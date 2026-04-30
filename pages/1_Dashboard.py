import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard · Olist", page_icon="📊", layout="wide")

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    st.warning("Please log in from the Home page.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]

st.title("📊 Sales Dashboard")
st.caption("Explore revenue trends, category performance, and customer behaviour.")
st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    all_cats = sorted(df["product_category_name"].unique().tolist())
    sel_cats = st.multiselect("Category", all_cats, default=all_cats[:5])
    all_status = sorted(df["order_status"].unique().tolist())
    sel_status = st.multiselect("Order Status", all_status, default=["delivered"])
    min_score, max_score = st.slider("Review Score", 1, 5, (1, 5))

filt = df.copy()
if sel_cats:
    filt = filt[filt["product_category_name"].isin(sel_cats)]
if sel_status:
    filt = filt[filt["order_status"].isin(sel_status)]
filt = filt[(filt["review_score"] >= min_score) & (filt["review_score"] <= max_score)]

if filt.empty:
    st.info("No data matches the current filters.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders (filtered)", f"{len(filt):,}")
k2.metric("Revenue",           f"R$ {filt['payment_value'].sum():,.0f}")
k3.metric("Avg Order Value",   f"R$ {filt['payment_value'].mean():.2f}")
k4.metric("Avg Review",        f"{filt['review_score'].mean():.2f}")

st.divider()

# ── Row 1: Revenue over time + Category bar ───────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    monthly = (
        filt.assign(Month=filt["order_purchase_timestamp"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["payment_value"].sum()
        .sort_values("Month")
    )
    fig = px.area(monthly, x="Month", y="payment_value",
                  title="Monthly Revenue", labels={"payment_value": "Revenue (R$)"},
                  color_discrete_sequence=["#1a6b3c"])
    fig.update_layout(margin=dict(t=40, b=20), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    cat_rev = (
        filt.groupby("product_category_name", as_index=False)["payment_value"].sum()
        .sort_values("payment_value", ascending=False)
        .head(10)
    )
    fig2 = px.bar(cat_rev, x="payment_value", y="product_category_name",
                  orientation="h", title="Top 10 Categories by Revenue",
                  labels={"payment_value": "Revenue (R$)", "product_category_name": "Category"},
                  color="payment_value", color_continuous_scale="Greens")
    fig2.update_layout(margin=dict(t=40, b=20), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Review distribution + Orders by DOW heatmap ────────────────────────
c3, c4 = st.columns(2)

with c3:
    review_counts = filt["review_score"].value_counts().sort_index().reset_index()
    review_counts.columns = ["Score", "Count"]
    fig3 = px.bar(review_counts, x="Score", y="Count",
                  title="Review Score Distribution",
                  color="Score", color_continuous_scale="RdYlGn")
    fig3.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    dow_hour = filt.copy()
    dow_hour["Hour"] = dow_hour["order_purchase_timestamp"].dt.hour
    dow_hour["DOW"]  = dow_hour["order_purchase_timestamp"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = dow_hour.groupby(["DOW","Hour"]).size().reset_index(name="Orders")
    heat["DOW"] = pd.Categorical(heat["DOW"], categories=dow_order, ordered=True)
    heat = heat.sort_values(["DOW","Hour"])
    pivot = heat.pivot(index="DOW", columns="Hour", values="Orders").fillna(0)
    fig4 = px.imshow(pivot, title="Orders by Day & Hour",
                     color_continuous_scale="Greens",
                     labels=dict(x="Hour of Day", y="Day of Week", color="Orders"))
    fig4.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Freight vs Price scatter ──────────────────────────────────────────
st.subheader("Price vs. Freight Value")
sample = filt.sample(min(500, len(filt)), random_state=1)
fig5 = px.scatter(sample, x="price", y="freight_value",
                  color="product_category_name", size="payment_value",
                  hover_data=["review_score", "order_status"],
                  opacity=0.7, title="Price vs Freight (sample 500 orders)")
fig5.update_layout(margin=dict(t=40, b=20))
st.plotly_chart(fig5, use_container_width=True)
