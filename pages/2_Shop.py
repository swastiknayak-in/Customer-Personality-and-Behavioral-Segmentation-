import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Shop", page_icon="🛒", layout="wide")

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

df:      pd.DataFrame = st.session_state["df"]
segment: str          = st.session_state.get("segment", "Regular")

def fmt_cat(s: str) -> str:
    return s.replace("_", " ").title()

# Emoji mapping per category
CAT_EMOJI = {
    "clothing_fashion":     "👗",
    "electronics":          "💻",
    "home_kitchen":         "🏠",
    "beauty_personal_care": "💄",
    "sports_fitness":       "🏋️",
    "books_stationery":     "📚",
    "toys_games":           "🎮",
    "groceries_food":       "🛒",
    "mobile_accessories":   "📱",
    "furniture_decor":      "🛋️",
    "health_wellness":      "💊",
    "automotive":           "🚗",
    "jewellery_watches":    "💍",
    "baby_products":        "🍼",
    "pet_supplies":         "🐾",
}

# Segment-based default price ceiling
PRICE_CEILINGS = {"Top": 50000, "Regular": 10000, "Budget": 2000}
default_max    = PRICE_CEILINGS.get(segment, 10000)

seg_label = {"Top": "⭐ Top Customer", "Regular": "🙂 Regular Customer",
             "Budget": "💡 Budget Shopper"}.get(segment, segment)

@st.cache_data
def build_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("category")
        .agg(
            avg_price      = ("price",           "mean"),
            avg_delivery   = ("delivery_charge", "mean"),
            avg_rating     = ("review_score",    "mean"),
            total_sold     = ("price",           "count"),
            avg_discount   = ("discount_amount", "mean"),
        )
        .reset_index()
    )
    for c in ["avg_price","avg_delivery","avg_rating","avg_discount"]:
        cat[c] = cat[c].round(2)
    cat["emoji"]        = cat["category"].map(CAT_EMOJI).fillna("📦")
    cat["display_name"] = cat["category"].apply(fmt_cat)
    return cat

catalogue = build_catalogue(df)

st.title("🛒 Shop")
st.caption(f"Welcome, **{seg_label}** — products are shown based on your shopping profile.")
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Search & Filter")
    search       = st.text_input("Search products", placeholder="e.g. electronics")
    price_max    = st.slider("Maximum Price (₹)", 49, 50000, default_max, step=100)
    min_rating   = st.slider("Minimum Rating ⭐", 1.0, 5.0, 3.5, 0.1)

filtered_cat = catalogue[
    (catalogue["avg_price"]  <= price_max) &
    (catalogue["avg_rating"] >= min_rating)
].copy()

if search:
    filtered_cat = filtered_cat[
        filtered_cat["display_name"].str.contains(search, case=False)
    ]

filtered_cat = filtered_cat.sort_values("avg_rating", ascending=False)

# ── Product grid ──────────────────────────────────────────────────────────────
if filtered_cat.empty:
    st.info("No products match your search. Try a different name or increase the price limit.")
else:
    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered_cat.iterrows()):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {row['emoji']} {row['display_name']}")

                rating_stars = "⭐" * round(row["avg_rating"])
                st.markdown(f"{rating_stars} **{row['avg_rating']:.1f}** out of 5")

                c1, c2 = st.columns(2)
                c1.markdown(f"**Price**")
                c1.markdown(f"₹ {row['avg_price']:,.0f}")
                c2.markdown(f"**Delivery**")
                c2.markdown(f"₹ {row['avg_delivery']:,.0f}")

                if row["avg_discount"] > 0:
                    st.markdown(
                        f"<span style='color:#10b981;font-weight:600'>"
                        f"Avg Discount: ₹ {row['avg_discount']:,.0f}</span>",
                        unsafe_allow_html=True,
                    )

                st.caption(f"{row['total_sold']:,} items sold")

                if st.button(
                    "Add to Cart", key=f"add_{idx}", use_container_width=True
                ):
                    st.session_state["cart"].append({
                        "name":  row["display_name"],
                        "emoji": row["emoji"],
                        "price": row["avg_price"],
                    })
                    st.success(f"{row['emoji']} Added to cart!")

st.divider()

# ── Cart ──────────────────────────────────────────────────────────────────────
cart = st.session_state["cart"]
st.subheader(f"🧾 Your Cart  ({len(cart)} item{'s' if len(cart) != 1 else ''})")

if not cart:
    st.info("Your cart is empty. Browse products above and add something!")
else:
    for i, item in enumerate(cart):
        col_name, col_price, col_del = st.columns([4, 1, 1])
        col_name.write(f"{item['emoji']}  {item['name']}")
        col_price.write(f"₹ {item['price']:,.0f}")
        if col_del.button("Remove", key=f"rm_{i}"):
            st.session_state["cart"].pop(i)
            st.rerun()

    total = sum(i["price"] for i in cart)
    st.markdown(f"### Total: ₹ {total:,.0f}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Cart", use_container_width=True):
            st.session_state["cart"] = []
            st.rerun()
    with col_b:
        if st.button("Place Order (Demo)", use_container_width=True):
            st.balloons()
            st.success("Order placed successfully! (This is a demo — no real transaction.)")
            st.session_state["cart"] = []

st.divider()

# ── Category popularity chart ─────────────────────────────────────────────────
st.subheader("📈 Most Popular Categories")
top12 = catalogue.sort_values("total_sold", ascending=False).head(12)
fig = px.bar(
    top12, x="display_name", y="total_sold",
    color="avg_rating",
    color_continuous_scale=["#c7d2fe", "#4F46E5"],
    labels={"total_sold": "Units Sold", "display_name": "Category",
            "avg_rating": "Avg Rating"},
    title="Top 12 Categories — Number of Items Sold",
    text_auto=True,
)
fig.update_xaxes(tickangle=-35)
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  margin=dict(t=45, b=20))
st.plotly_chart(fig, use_container_width=True)
