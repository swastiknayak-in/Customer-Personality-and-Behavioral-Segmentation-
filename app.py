import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.db_handler   import init_db, verify_user
from utils.mock_data    import generate_mock_data
from utils.ml_engine    import preprocess_data, train_rfm_segments, train_churn_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Retail Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme injection ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Accent colour: deep indigo #4F46E5 with violet highlights */
[data-testid="stSidebar"] { background: #1e1b4b; }
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #4F46E5; color: #fff; border-radius: 8px; border: none; }
[data-testid="stSidebar"] .stButton > button:hover { background: #6366f1; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#4F46E5,#7c3aed);
    border-radius: 12px; padding: 16px; color: #fff; }
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div { color: #fff !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
h1,h2,h3 { color: #312e81; }
</style>
""", unsafe_allow_html=True)

init_db()

# ── Session defaults ──────────────────────────────────────────────────────────
for key, default in {
    "logged_in": False, "segment": "Regular",
    "cart": [], "df": None, "rfm": None,
    "churn_model": None, "churn_features": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data, please wait…")
def load_data():
    try:
        df_raw    = generate_mock_data(n_rows=3000)
        clean     = preprocess_data(df_raw)
        rfm       = train_rfm_segments(clean)
        model, ft = train_churn_model(clean)
        return clean, rfm, model, ft
    except Exception as e:
        st.warning(f"Something went wrong loading data: {e}")
        df_raw    = generate_mock_data(n_rows=1000)
        clean     = preprocess_data(df_raw)
        rfm       = train_rfm_segments(clean)
        model, ft = train_churn_model(clean)
        return clean, rfm, model, ft


if st.session_state["df"] is None:
    df, rfm, model, ft = load_data()
    st.session_state.update({
        "df": df, "rfm": rfm,
        "churn_model": model, "churn_features": ft,
    })


# ── LOGIN ─────────────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown("""
    <style>[data-testid="stSidebar"]{display:none}</style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown("## 🛍️ Smart Retail Platform")
        st.markdown("#### Sign in to your account")

        st.info(
            "**Demo Accounts**\n\n"
            "| Account | Email | Password |\n"
            "|---------|-------|----------|\n"
            "| Admin | admin@demo.com | admin123 |\n"
            "| User  | user@demo.com  | user123  |\n"
            "| Guest | guest@demo.com | guest123 |"
        )

        with st.form("login"):
            email = st.text_input("Email address", placeholder="you@example.com")
            pwd   = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                seg = verify_user(email, pwd)
                if seg:
                    st.session_state["logged_in"] = True
                    st.session_state["segment"]   = seg
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again.")

# ── HOME (after login) ────────────────────────────────────────────────────────
else:
    seg       = st.session_state["segment"]
    seg_label = {"Top": "⭐ Top Customer", "Regular": "🙂 Regular Customer",
                 "Budget": "💡 Budget Shopper"}.get(seg, seg)

    with st.sidebar:
        st.markdown(f"### {seg_label}")
        st.markdown("---")
        st.page_link("app.py",               label="🏠  Home")
        st.page_link("pages/1_Dashboard.py", label="📊  Dashboard")
        st.page_link("pages/2_Shop.py",      label="🛒  Shop")
        st.page_link("pages/3_Insights.py",  label="🔍  Customer Insights")
        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            st.session_state.update({"logged_in": False, "segment": "Regular", "cart": []})
            st.rerun()

    st.markdown("# 🛍️ Smart Retail Platform")
    st.caption("Your complete view of sales, products, and customer behaviour")
    st.divider()

    df = st.session_state["df"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders",       f"{len(df):,}")
    k2.metric("Unique Customers",   f"{df['customer_id'].nunique():,}")
    k3.metric("Total Revenue",      f"₹{df['payment_amount'].sum():,.0f}")
    k4.metric("Average Rating",     f"{df['review_score'].mean():.1f} / 5")

    st.divider()
    st.markdown("""
    ### Where would you like to go?

    | Page | What you can do |
    |------|-----------------|
    | 📊 **Dashboard** | View sales trends, top categories, and order patterns |
    | 🛒 **Shop** | Browse and add products to your cart |
    | 🔍 **Customer Insights** | See customer groups and buying risk analysis |
    """)
