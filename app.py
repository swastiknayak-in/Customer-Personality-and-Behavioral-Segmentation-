"""
ShopIndia — Customer Segmentation & Behaviour Analysis Platform
=================================================================
Single-run Streamlit app.  All heavy computation is cached once.

Portals
  Customer  →  personalized shop, order history, behaviour profile
  Manager   →  segmentation, behaviour analysis, P&L, full analytics
"""

import streamlit as st
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.auth import init_db, verify
from utils.data import (
    generate_data, compute_rfm, train_churn_model,
    compute_behavior, compute_pnl,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopIndia Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 60%, #24243e 100%);
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #fff !important; font-weight: 600; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
    border: none;
}
[data-testid="metric-container"] * { color: #fff !important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 600 !important; opacity: .8; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff; border: none; border-radius: 10px;
    font-weight: 600; padding: 10px 24px;
    transition: all .2s; box-shadow: 0 3px 14px rgba(102,126,234,.4);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,.5); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f0f0f8; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#667eea,#764ba2); color: #fff !important; }

/* Cards */
div.card {
    background: #fff; border-radius: 16px;
    border: 1px solid #e8e4f0;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.05);
}

/* Page headers */
h1 { font-weight: 800 !important; }
h2 { font-weight: 700 !important; }
h3 { font-weight: 600 !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Success / info boxes */
.stSuccess { border-radius: 10px; }
.stInfo    { border-radius: 10px; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DB + session defaults ─────────────────────────────────────────────────────
init_db()
for k, v in {"logged_in": False, "role": None, "name": None,
             "data_loaded": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── One-time data computation (cached globally) ───────────────────────────────
@st.cache_data(show_spinner="⚡ Loading data and training models (one-time only)…")
def load_all():
    df      = generate_data(n_rows=4000)
    rfm     = compute_rfm(df)
    model, feats = train_churn_model(rfm)
    behavior     = compute_behavior(df)
    pnl          = compute_pnl(df)
    return df, rfm, model, feats, behavior, pnl

# Load once and store in session
if not st.session_state["data_loaded"]:
    (st.session_state["df"],
     st.session_state["rfm"],
     st.session_state["churn_model"],
     st.session_state["churn_feats"],
     st.session_state["behavior"],
     st.session_state["pnl"]) = load_all()
    st.session_state["data_loaded"] = True


# ── Login screen ──────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown("""
    <style>[data-testid="stSidebar"]{display:none}</style>
    <div style='text-align:center;padding:40px 0 10px'>
        <div style='font-size:52px'>🛍️</div>
        <h1 style='background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-size:36px;margin-bottom:4px'>ShopIndia Intelligence</h1>
        <p style='color:#666;font-size:16px'>Customer Segmentation & Behaviour Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("---")
        st.markdown("#### Sign in to your portal")
        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@example.com")
            pwd   = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            result = verify(email, pwd)
            if result:
                st.session_state["logged_in"] = True
                st.session_state["role"]      = result[0]
                st.session_state["name"]      = result[1]
                st.rerun()
            else:
                st.error("Invalid email or password.")

        st.markdown("""
        <div style='background:#f8f6ff;border-radius:12px;padding:14px 18px;margin-top:16px;border:1px solid #e0d8f8'>
        <b>Demo Accounts</b><br>
        <small>
        👤 <b>Customer:</b> customer@demo.com / customer123<br>
        📊 <b>Manager:</b> manager@demo.com / manager123
        </small></div>
        """, unsafe_allow_html=True)
    st.stop()


# ── Routed portals ────────────────────────────────────────────────────────────
role = st.session_state["role"]
name = st.session_state["name"]

with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 10px'>
        <div style='font-size:28px;text-align:center'>🛍️</div>
        <div style='text-align:center;font-weight:800;font-size:17px;color:#fff;margin-top:6px'>ShopIndia</div>
        <div style='text-align:center;font-size:12px;color:#aaa;margin-top:2px'>Intelligence Platform</div>
    </div>
    <div style='background:rgba(255,255,255,.08);border-radius:10px;padding:10px 14px;margin-bottom:18px'>
        <div style='font-weight:600;color:#fff;font-size:14px'>{name}</div>
        <div style='font-size:12px;color:#aaa'>{"👤 Customer" if role=="customer" else "📊 Manager"}</div>
    </div>
    """, unsafe_allow_html=True)

    if role == "customer":
        page = st.radio("Navigate", ["🏠 My Home", "🛒 Shop", "📦 My Orders", "👤 My Profile"], label_visibility="collapsed")
    else:
        page = st.radio("Navigate", [
            "📊 Dashboard",
            "👥 Customer Segmentation",
            "🧠 Behaviour Analysis",
            "💰 Profit & Loss",
            "📈 Advanced Analytics",
            "⚠️ Churn Risk",
            "🛒 Product Analytics",
        ], label_visibility="collapsed")

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        for k in ["logged_in", "role", "name"]:
            st.session_state[k] = False if k == "logged_in" else None
        st.rerun()


# ── Import and run the correct portal ────────────────────────────────────────
if role == "customer":
    from pages.customer_portal import run_customer
    run_customer(page, st.session_state["df"], st.session_state["rfm"],
                 st.session_state["behavior"], name)
else:
    from pages.manager_portal import run_manager
    run_manager(page, st.session_state["df"], st.session_state["rfm"],
                st.session_state["churn_model"], st.session_state["churn_feats"],
                st.session_state["behavior"], st.session_state["pnl"])
