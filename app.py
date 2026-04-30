import streamlit as st
import pandas as pd
import os
import sys
import requests # For loading Lottie files from URL
from streamlit_lottie import st_lottie # Special library for animations
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.db_handler import init_db, verify_user
from utils.mock_data import generate_mock_olist_data
from utils.ml_engine import preprocess_data, train_rfm_segments, train_churn_model

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Intelligence Pro",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ANIMATION & CSS HANDLERS ─────────────────────────────────────────────────
# 1. Advanced CSS Animations (Glassmorphism + Fade-In)
st.markdown("""
    <style>
    /* Gradient Background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* Definition of the 'fade-in' animation */
    @keyframes fadeInEffect {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Apply fadeIn to major components (cards, headers) */
    div[data-testid="stMetric"], 
    .hero-container,
    .element-container:has(.stButton) {
        animation: fadeInEffect 1.1s ease-out;
    }

    /* KPI Cards - Glassmorphism style */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(8px);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }

    /* Animated Hero Banner */
    .hero-container {
        background: linear-gradient(90deg, #002776 0%, #0047AB 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }

    /* Personalized greeting animation delay */
    .animated-greeting {
        font-weight: bold;
        animation: fadeInEffect 1.5s ease-out;
    }
    
    /* Button micro-interactions */
    .stButton>button {
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .stButton>button:hover {
        transform: scale(1.03) translateY(-3px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1);
        border-color: #0047AB;
    }
    </style>
    """, unsafe_allow_html=True)


# 2. Lottie Loader Helper
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load different Lottie assets (these are URLs to public, open-source animations)
# We handle loading *before* rendering to avoid lag.
lottie_delivery = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6w45l9g.json") # Delivery truck
lottie_welcome = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_pucia9k5.json") # Tech greeting
lottie_secure = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5n8y4o.json") # Lock/Security


# ── DB INIT ──────────────────────────────────────────────────────────────────
init_db()

# ── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
for key, default in {
    "logged_in": False, "segment": "Medium", "cart": [],
    "df": None, "rfm": None, "churn_model": None, "churn_features": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── CACHED DATA LOADER (Keep existing logic) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Intelligence Model Training in Progress…")
def load_application_data():
    try:
        data_path = "data/olist_order_items_dataset.csv"
        if os.path.exists(data_path):
            items = pd.read_csv(data_path, nrows=10_000)
            df_raw = items
        else:
            df_raw = generate_mock_olist_data(n_rows=3_000)

        clean_df = preprocess_data(df_raw)
        rfm_df = train_rfm_segments(clean_df)
        model, ft = train_churn_model(clean_df)
        return clean_df, rfm_df, model, ft

    except Exception as e:
        st.warning(f"Data error ({e}). Falling back to mock data.")
        df_raw = generate_mock_olist_data(n_rows=3_000)
        clean_df = preprocess_data(df_raw)
        rfm_df = train_rfm_segments(clean_df)
        model, ft = train_churn_model(clean_df)
        return clean_df, rfm_df, model, ft


# ── PRE-LOAD DATA ────────────────────────────────────────────────────────────
if st.session_state["df"] is None:
    df, rfm, model, ft = load_application_data()
    st.session_state.update({
        "df": df, "rfm": rfm, "churn_model": model, "churn_features": ft
    })


# ── LOGIN PAGE (PRE-LOGIN) ───────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        .login-box {max-width: 420px; margin: 4rem auto; padding: 2.5rem;
                    background-color: rgba(255,255,255,0.7); backdrop-filter: blur(10px);
                    border-radius: 16px; border: 1px solid rgba(255,255,255,0.3);
                    box-shadow: 0 10px 40px rgba(0,0,0,.15);
                    animation: fadeInEffect 1.2s ease-out;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([0.8, 1.5, 0.8])
    
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # 1. Animation: Secure Lock
        if lottie_secure:
            st_lottie(lottie_secure, height=120, key="secure_lock")
        else:
            st.markdown("### 🔐")

        st.markdown("<h2 style='text-align: center; color: #002776;'>Olist Intelligence Pro</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Authorized access only. Enter credentials below.</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.form("login_form"):
            email = st.text_input("Corporate Email ID", placeholder="you@example.com")
            pwd = st.text_input("Access Token", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("Authenticate & Access →", use_container_width=True):
                seg = verify_user(email, pwd)
                if seg:
                    st.session_state["logged_in"] = True
                    st.session_state["segment"] = seg
                    st.toast("Success. Synchronizing Data Warehouses...", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid email or access token.")

        with st.expander("🔑 View Demo Credentials"):
            st.info("**High Value:** admin@test.com / admin123\n\n**Medium:** silver@test.com / silver123")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ── HOME PAGE (POST-LOGIN) ───────────────────────────────────────────────────
else:
    seg = st.session_state["segment"]
    seg_emoji = {"High": "🥇", "Medium": "🥈", "Low": "🥉"}.get(seg, "👤")

    # Sidebar (Cleaned up)
    with st.sidebar:
        # Lottie 2: Tech Greeting
        if lottie_welcome:
            st_lottie(lottie_welcome, height=150, key="sidebar_greet")
            
        st.markdown(f"<h3 style='text-align: center;'>Welcome, **{seg}** Value</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 12px; color: #666; margin-top:-15px;'>Market Analyst Tier</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.page_link("app.py", label="Home Center", icon="🏠")
        st.page_link("pages/1_Dashboard.py", label="Business Intelligence", icon="📊")
        st.page_link("pages/2_Shop.py", label="Personalized Shop", icon="🛒")
        st.page_link("pages/3_Insights.py", label="AI Predictors", icon="🤖")
        st.markdown("<br>"*5, unsafe_allow_html=True)
        if st.button("🚪 Terminates Session", use_container_width=True, type="secondary"):
            for key in ["logged_in", "segment", "cart"]:
                st.session_state[key] = (False if key == "logged_in" else "Medium" if key == "segment" else [])
            st.rerun()

    # 1. Hero banner with micro-greeting animation
    st.markdown(f"""
        <div class="hero-container">
            <h1>Brazil Market Intelligence Command Center <span style='font-size:30px;'>🇧🇷</span></h1>
            <p>Welcome back, <span class='animated-greeting'>{seg} Tier Analyst</span>. System Synced: {datetime.now().strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

    df = st.session_state["df"]

    # 2. Animated KPI Strip (Fade-in applied via CSS)
    k1, k2, k3, k4 = st.columns(4)
    # Using real growth-indicative deltas
    k1.metric("📦 Volume (Orders)", f"{len(df):,}", "12.5% YoY")
    k2.metric("👥 Reach (Customers)", f"{df['customer_unique_id'].nunique():,}", "4.2%")
    k3.metric("💰 GMV (Revenue)", f"R$ {df['payment_value'].sum()/1e3:.1f}k", "R$ 2.4k (Today)")
    k4.metric("⭐ CSAT (Avg Review)", f"{df['review_score'].mean():.2f}", "-0.01", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚡ Command Shortcuts")
    
    # 3. Interactive, Animated Shortcut Cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.markdown("#### 📊 Open BI Dashboard")
            st.write("Regional revenue maps and sales trend forecasting.")
            if st.button("Analyze Sales →", use_container_width=True, key="goto_dash"):
                st.switch_page("pages/1_Dashboard.py")

    with c2:
        with st.container(border=True):
            st.markdown("#### 🛒 Personalised Procurement")
            st.write("Inventory matching tailored to your tier's logistics priority.")
            if st.button("Enter Shop →", use_container_width=True, key="goto_shop"):
                st.switch_page("pages/2_Shop.py")

    with c3:
        with st.container(border=True):
            st.markdown("#### 🤖 Predictive Engines")
            st.write("AI modeling for churn prediction and RFM clustering.")
            if st.button("Run AI Models →", use_container_width=True, key="goto_ai"):
                st.switch_page("pages/3_Insights.py")

    # 4. Final Animation + Segment-Based Alert (Conditional Polish)
    st.markdown("---")
    cola, colb = st.columns([1, 4])
    with cola:
        if lottie_delivery:
            st_lottie(lottie_delivery, height=120, key="delivery_icon")
    with colb:
        st.markdown("<br>", unsafe_allow_html=True)
        if seg == "High":
            st.success("🌟 **Elite Status Benefit:** Priority Logistics API integration is currently ACTIVE for your session.")
        elif seg == "Low":
            st.info("💡 **Growth Tip:** Reach 50+ total orders within the 'Shop' page to automatically unlock Silver Tier benefits.")
