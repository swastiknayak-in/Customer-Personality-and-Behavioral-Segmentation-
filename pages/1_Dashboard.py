import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e1b4b; }
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #4F46E5; color: #fff; border-radius: 8px; border: none; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#4F46E5,#7c3aed);
    border-radius: 12px; padding: 16px; }
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div { color: #fff !important; }
h1,h2,h3 { color: #312e81; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("logged_in"):
    st.warning("Please sign in from the Home page first.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]

# ── Helper: clean category display name ──────────────────────────────────────
def fmt_cat(s: str) -> str:
    return s.replace("_", " ").title()

st.title("📊 Sales Dashboard")
st.caption("Track your store's performance — revenue, categories, and customer activity.")
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Filters")

    all_cats = sorted(df["category"].unique().tolist())
    sel_cats = st.multiselect(
        "Product Category",
        options=all_cats,
        default=all_cats[:6],
        format_func=fmt_cat,
    )

    all_cities = sorted(df["city"].unique().tolist())
    sel_cities = st.multiselect("City", options=all_cities, default=all_cities)

    sel_status = st.multiselect(
        "Order Status",
        options=sorted(df["order_status"].unique()),
        default=["delivered"],
        format_func=str.title,
    )

    min_score, max_score = st.slider("Rating Range", 1, 5, (1, 5))

filt = df.copy()
if sel_cats:    filt = filt[filt["category"].isin(sel_cats)]
if sel_cities:  filt = filt[filt["city"].isin(sel_cities)]
if sel_status:  filt = filt[filt["order_status"].isin(sel_status)]
filt = filt[(filt["review_score"] >= min_score) & (filt["review_score"] <= max_score)]

if filt.empty:
    st.info("No orders match the selected filters. Try changing your selections.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders",          f"{len(filt):,}")
k2.metric("Total Revenue",   f"₹{filt['payment_amount'].sum():,.0f}")
k3.metric("Avg Order Value", f"₹{filt['payment_amount'].mean():,.0f}")
k4.metric("Avg Rating",      f"{filt['review_score'].mean():.1f} ⭐")

st.divider()

PALETTE = ["#4F46E5", "#7c3aed", "#a855f7", "#ec4899", "#f59e0b",
           "#10b981", "#06b6d4", "#f97316", "#84cc16", "#e11d48"]

# ── Row 1: Monthly Revenue + Top Categories ───────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    monthly = (
        filt.assign(Month=filt["order_date"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["payment_amount"].sum()
        .sort_values("Month")
    )
    fig = px.area(
        monthly, x="Month", y="payment_amount",
        title="Monthly Revenue (₹)",
        labels={"payment_amount": "Revenue (₹)", "Month": "Month"},
        color_discrete_sequence=["#4F46E5"],
    )
    fig.update_layout(margin=dict(t=45, b=20), xaxis_tickangle=-40,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(fillcolor="rgba(79,70,229,0.15)")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    cat_rev = (
        filt.groupby("category", as_index=False)["payment_amount"].sum()
        .sort_values("payment_amount", ascending=False)
        .head(10)
    )
    cat_rev["Category"] = cat_rev["category"].apply(fmt_cat)
    fig2 = px.bar(
        cat_rev, x="payment_amount", y="Category", orientation="h",
        title="Top 10 Categories by Revenue",
        labels={"payment_amount": "Revenue (₹)"},
        color="payment_amount",
        color_continuous_scale=["#c7d2fe", "#4F46E5"],
    )
    fig2.update_layout(margin=dict(t=45, b=20), yaxis=dict(autorange="reversed"),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Rating Distribution + Orders by Day & Hour ────────────────────────
c3, c4 = st.columns(2)

with c3:
    rc = filt["review_score"].value_counts().sort_index().reset_index()
    rc.columns = ["Rating", "Number of Orders"]
    fig3 = px.bar(
        rc, x="Rating", y="Number of Orders",
        title="Customer Ratings Breakdown",
        color="Rating",
        color_continuous_scale=["#ef4444", "#f59e0b", "#4F46E5", "#10b981", "#059669"],
        text_auto=True,
    )
    fig3.update_layout(margin=dict(t=45, b=20),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    dh = filt.copy()
    dh["Hour"] = dh["order_date"].dt.hour
    dh["Day"]  = dh["order_date"].dt.day_name()
    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = dh.groupby(["Day","Hour"]).size().reset_index(name="Orders")
    heat["Day"] = pd.Categorical(heat["Day"], categories=day_order, ordered=True)
    pivot = heat.pivot(index="Day", columns="Hour", values="Orders").fillna(0)
    fig4 = px.imshow(
        pivot, title="When Do Customers Shop?",
        color_continuous_scale=["#e0e7ff", "#4F46E5"],
        labels=dict(x="Hour of Day", y="", color="Orders"),
    )
    fig4.update_layout(margin=dict(t=45, b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: City-wise Revenue + Order Status Pie ───────────────────────────────
c5, c6 = st.columns(2)

with c5:
    city_rev = (
        filt.groupby("city", as_index=False)["payment_amount"].sum()
        .sort_values("payment_amount", ascending=False)
    )
    fig5 = px.bar(
        city_rev, x="city", y="payment_amount",
        title="Revenue by City",
        labels={"city": "City", "payment_amount": "Revenue (₹)"},
        color="payment_amount",
        color_continuous_scale=["#c7d2fe", "#7c3aed"],
        text_auto=".2s",
    )
    fig5.update_layout(margin=dict(t=45, b=20),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    status_ct = filt["order_status"].str.title().value_counts().reset_index()
    status_ct.columns = ["Status", "Count"]
    fig6 = px.pie(
        status_ct, names="Status", values="Count",
        title="Order Status Breakdown",
        color_discrete_sequence=["#4F46E5","#10b981","#f59e0b","#ef4444"],
        hole=0.45,
    )
    fig6.update_layout(margin=dict(t=45, b=10))
    st.plotly_chart(fig6, use_container_width=True)

# ── Row 4: Price vs Delivery Charge scatter ───────────────────────────────────
st.subheader("Product Price vs Delivery Charge")
st.caption("Each dot is one order. Larger dots = higher total payment.")
sample = filt.sample(min(600, len(filt)), random_state=1)
sample["Category"] = sample["category"].apply(fmt_cat)
fig7 = px.scatter(
    sample, x="price", y="delivery_charge",
    color="Category", size="payment_amount",
    hover_data={"review_score": True, "order_status": True, "city": True},
    opacity=0.65,
    title="Price vs Delivery Charge",
    labels={"price": "Product Price (₹)", "delivery_charge": "Delivery Charge (₹)"},
    color_discrete_sequence=PALETTE,
)
fig7.update_layout(margin=dict(t=45, b=20),
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig7, use_container_width=True)
