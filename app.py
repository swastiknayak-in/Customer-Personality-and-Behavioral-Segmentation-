import streamlit as st
import pandas as pd
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.db_handler import init_db, verify_user
from utils.mock_data import generate_mock_olist_data
from utils.ml_engine import preprocess_data, train_rfm_segments, train_churn_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Intelligence Platform",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB init ──────────────────────────────────────────────────────────────────
init_db()

# ── Session state defaults ───────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "segment":   "Medium",
    "cart":      [],
    "df":        None,
    "rfm":       None,
    "churn_model": None,
    "churn_features": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Data loader (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading data & training models…")
def load_application_data():
    try:
        data_path = "data/olist_order_items_dataset.csv"
        if os.path.exists(data_path):
            items = pd.read_csv(data_path, nrows=10_000)
            df_raw = items
        else:
            df_raw = generate_mock_olist_data(n_rows=3_000)

        clean_df  = preprocess_data(df_raw)
        rfm_df    = train_rfm_segments(clean_df)
        model, ft = train_churn_model(clean_df)
        return clean_df, rfm_df, model, ft

    except Exception as e:
        st.warning(f"Data error ({e}). Falling back to mock data.")
        df_raw    = generate_mock_olist_data(n_rows=3_000)
        clean_df  = preprocess_data(df_raw)
        rfm_df    = train_rfm_segments(clean_df)
        model, ft = train_churn_model(clean_df)
        return clean_df, rfm_df, model, ft


# ── Load & store in session ──────────────────────────────────────────────────
if st.session_state["df"] is None:
    df, rfm, model, ft = load_application_data()
    st.session_state["df"]             = df
    st.session_state["rfm"]            = rfm
    st.session_state["churn_model"]    = model
    st.session_state["churn_features"] = ft


# ── LOGIN PAGE ───────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        .login-box {max-width: 420px; margin: 6rem auto; padding: 2rem;
                    border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,.12);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔐 Olist Platform Login")
        st.info(
            "**Demo credentials**\n\n"
            "| Role | Email | Password |\n"
            "|------|-------|----------|\n"
            "| High Value | admin@test.com | admin123 |\n"
            "| Medium Value | silver@test.com | silver123 |\n"
            "| Low Value | bronze@test.com | bronze123 |"
        )

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            pwd   = st.text_input("Password", type="password")
            if st.form_submit_button("Login →", use_container_width=True):
                seg = verify_user(email, pwd)
                if seg:
                    st.session_state["logged_in"] = True
                    st.session_state["segment"]   = seg
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password.")

# ── HOME PAGE (post-login) ───────────────────────────────────────────────────
else:
    seg = st.session_state["segment"]
    seg_emoji = {"High": "🥇", "Medium": "🥈", "Low": "🥉"}.get(seg, "👤")

    with st.sidebar:
        st.markdown(f"### {seg_emoji} {seg} Value User")
        st.markdown("---")
        st.page_link("app.py",            label="🏠 Home")
        st.page_link("pages/1_Dashboard.py", label="📊 Dashboard")
        st.page_link("pages/2_Shop.py",      label="🛒 Shop")
        st.page_link("pages/3_Insights.py",  label="🤖 ML Insights")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "segment", "cart"]:
                st.session_state[key] = (False if key == "logged_in" else
                                         "Medium" if key == "segment" else [])
            st.rerun()

    # ── Hero section ─────────────────────────────────────────────────────────
    st.markdown("# 🇧🇷 Brazilian E-Commerce Intelligence Platform")
    st.caption("Powered by the Olist dataset · Built with Streamlit")
    st.divider()

    df  = st.session_state["df"]
    rfm = st.session_state["rfm"]

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📦 Total Orders",    f"{len(df):,}")
    k2.metric("👥 Unique Customers", f"{df['customer_unique_id'].nunique():,}")
    k3.metric("💰 Total Revenue",   f"R$ {df['payment_value'].sum():,.0f}")
    k4.metric("⭐ Avg Review",       f"{df['review_score'].mean():.2f} / 5")

    st.divider()
    st.markdown(
        """
        ### 👈 Use the sidebar to explore:
        | Page | Description |
        |------|-------------|
        | 📊 **Dashboard** | Sales trends, category breakdown, order heatmaps |
        | 🛒 **Shop** | Browse products personalised to your segment |
        | 🤖 **ML Insights** | RFM clustering, churn prediction, feature importance |
        """
    )
