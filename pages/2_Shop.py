import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Shop · Olist", page_icon="🛒", layout="wide")

if not st.session_state.get("logged_in"):
    st.warning("Please log in from the Home page.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
segment: str      = st.session_state.get("segment", "Medium")
cart: list        = st.session_state.get("cart", [])

# ── Segment-based price range ─────────────────────────────────────────────────
PRICE_RANGES = {
    "High":   (200,  2500),
    "Medium": (50,   500),
    "Low":    (5,    100),
}
p_min, p_max = PRICE_RANGES.get(segment, (5, 2500))

# ── Build product catalogue from dataset ──────────────────────────────────────
@st.cache_data
def build_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("product_category_name")
        .agg(
            avg_price   = ("price", "mean"),
            avg_review  = ("review_score", "mean"),
            total_sold  = ("price", "count"),
            avg_freight = ("freight_value", "mean"),
        )
        .reset_index()
        .rename(columns={"product_category_name": "category"})
    )
    cat["avg_price"]   = cat["avg_price"].round(2)
    cat["avg_freight"] = cat["avg_freight"].round(2)
    cat["avg_review"]  = cat["avg_review"].round(2)
    np.random.seed(99)
    cat["emoji"] = np.random.choice(
        ["📦","🛋️","💄","👟","🔧","💻","🧸","🌿","🚗","📚","🍕","🐾","🎮","⌚","🏋️"],
        size=len(cat), replace=True,
    )
    return cat

catalogue = build_catalogue(df)

st.title("🛒 Personalised Shop")
st.caption(f"Showing recommendations for **{segment} Value** customers · R$ {p_min}–{p_max} range")
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    price_filter = st.slider("Max Price (R$)", int(df["price"].min()), int(df["price"].max()),
                             p_max, step=10)
    min_review = st.slider("Minimum Review ⭐", 1.0, 5.0, 3.5, 0.1)
    search = st.text_input("Search category", "")

# Apply filters
filtered_cat = catalogue[
    (catalogue["avg_price"] <= price_filter) &
    (catalogue["avg_review"] >= min_review)
]
if search:
    filtered_cat = filtered_cat[filtered_cat["category"].str.contains(search, case=False)]

filtered_cat = filtered_cat.sort_values("avg_review", ascending=False)

# ── Product grid ──────────────────────────────────────────────────────────────
if filtered_cat.empty:
    st.info("No products match the current filters. Try relaxing the criteria.")
else:
    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered_cat.iterrows()):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {row['emoji']} {row['category'].replace('_', ' ').title()}")
                st.markdown(f"**Price:** R$ {row['avg_price']:.2f}")
                st.markdown(f"**Shipping:** R$ {row['avg_freight']:.2f}")
                st.markdown(f"**Rating:** {'⭐' * round(row['avg_review'])} ({row['avg_review']:.1f})")
                st.markdown(f"**Sold:** {row['total_sold']:,} units")

                if st.button("🛒 Add to Cart", key=f"add_{idx}", use_container_width=True):
                    item = {
                        "category": row["category"],
                        "price":    row["avg_price"],
                        "emoji":    row["emoji"],
                    }
                    st.session_state["cart"].append(item)
                    st.success(f"Added {row['emoji']} to cart!")

st.divider()

# ── Cart section ──────────────────────────────────────────────────────────────
st.subheader(f"🧾 Cart ({len(st.session_state['cart'])} items)")
if not st.session_state["cart"]:
    st.info("Your cart is empty. Add some products above!")
else:
    cart_df = pd.DataFrame(st.session_state["cart"])
    total   = cart_df["price"].sum()

    for i, item in enumerate(st.session_state["cart"]):
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"{item['emoji']} {item['category'].replace('_', ' ').title()}")
        c2.write(f"R$ {item['price']:.2f}")
        if c3.button("❌", key=f"remove_{i}"):
            st.session_state["cart"].pop(i)
            st.rerun()

    st.markdown(f"**Total: R$ {total:.2f}**")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Cart", use_container_width=True):
            st.session_state["cart"] = []
            st.rerun()
    with col_b:
        if st.button("✅ Checkout (Demo)", use_container_width=True):
            st.balloons()
            st.success("🎉 Order placed! (This is a demo — no real transaction.)")
            st.session_state["cart"] = []

# ── Category popularity chart ─────────────────────────────────────────────────
st.divider()
st.subheader("📈 Category Popularity")
top_cats = catalogue.sort_values("total_sold", ascending=False).head(12)
fig = px.bar(top_cats, x="category", y="total_sold",
             color="avg_review", color_continuous_scale="Blues",
             labels={"total_sold": "Units Sold", "category": "Category", "avg_review": "Avg Review"},
             title="Top 12 Categories by Sales Volume")
fig.update_xaxes(tickangle=-35)
st.plotly_chart(fig, use_container_width=True)
