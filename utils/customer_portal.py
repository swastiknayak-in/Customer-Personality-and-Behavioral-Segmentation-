"""Customer Portal — Shop, Orders, Profile, Behaviour Summary."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

PURPLE = "#667eea"
VIOLET = "#764ba2"
SEQ    = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#fda085", "#4facfe", "#43e97b", "#fa709a"]

def fmt(n): return f"₹{n:,.0f}"


def run_customer(page, df, rfm, behavior, name):
    # Simulate "this customer" = first customer
    my_id  = df["customer_id"].iloc[0]
    my_ord = df[df["customer_id"] == my_id]
    my_rfm = rfm[rfm["customer_id"] == my_id]
    my_beh = behavior[behavior["customer_id"] == my_id]

    seg = my_rfm["Segment"].values[0] if len(my_rfm) else "Regular Customers"
    seg_emoji = {"Top Customers":"🥇","Loyal Customers":"🥈","Regular Customers":"🥉","Budget Buyers":"💡"}.get(seg,"👤")

    # ── HOME ──────────────────────────────────────────────────────────────────
    if "Home" in page:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);border-radius:20px;
             padding:32px 36px;margin-bottom:28px'>
            <h1 style='color:#fff;margin:0;font-size:28px'>Welcome back, {name}! 👋</h1>
            <p style='color:rgba(255,255,255,.75);margin-top:6px;font-size:15px'>
                {seg_emoji} You are a <b>{seg}</b> — thank you for being with us!</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        total_ord = len(my_ord)
        total_spe = my_ord["payment_amount"].sum()
        avg_rat   = my_ord["review_score"].mean()
        disc_save = my_ord["discount_amount"].sum()
        c1.metric("Total Orders",   total_ord)
        c2.metric("Total Spent",    fmt(total_spe))
        c3.metric("Avg Rating Given", f"{avg_rat:.1f} ⭐")
        c4.metric("Savings via Discounts", fmt(disc_save))

        st.markdown("---")
        st.subheader("📅 Your Spending Over Time")
        monthly = (my_ord.groupby("month_name")["payment_amount"].sum()
                   .reset_index().rename(columns={"payment_amount":"Spent"}))
        fig = px.area(monthly, x="month_name", y="Spent",
                      color_discrete_sequence=[PURPLE], title="Monthly Spending (₹)")
        fig.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🛍️ Favourite Categories")
            cat_spe = my_ord.groupby("category")["payment_amount"].sum().reset_index()
            fig2 = px.pie(cat_spe, names="category", values="payment_amount",
                          color_discrete_sequence=SEQ, hole=0.42)
            fig2.update_layout(showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("📦 Order Status")
            s = my_ord["order_status"].value_counts().reset_index()
            s.columns = ["Status", "Count"]
            fig3 = px.bar(s, x="Status", y="Count", color="Status",
                          color_discrete_sequence=SEQ, text_auto=True)
            fig3.update_layout(showlegend=False, plot_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

    # ── SHOP ──────────────────────────────────────────────────────────────────
    elif "Shop" in page:
        st.markdown(f"## 🛒 Shop — Personalized for {seg_emoji} {seg}")
        st.caption("Products and deals curated based on your purchase history and segment.")

        cats = sorted(df["category"].unique())
        sel_cat = st.selectbox("Filter by Category", ["All"] + cats)
        col_s, col_p = st.columns([3, 1])
        search = col_s.text_input("Search products", placeholder="Type to search…")
        sort_by = col_p.selectbox("Sort by", ["Popularity", "Price ↑", "Price ↓", "Rating"])

        prod_df = (df.groupby("category").agg(
            avg_price    = ("price", "mean"),
            avg_rating   = ("review_score", "mean"),
            total_orders = ("order_date", "count"),
            avg_discount = ("discount_pct", "mean"),
        ).reset_index())

        if sel_cat != "All":
            prod_df = prod_df[prod_df["category"] == sel_cat]
        if search:
            prod_df = prod_df[prod_df["category"].str.lower().str.contains(search.lower())]

        if sort_by == "Price ↑":   prod_df = prod_df.sort_values("avg_price")
        elif sort_by == "Price ↓": prod_df = prod_df.sort_values("avg_price", ascending=False)
        elif sort_by == "Rating":  prod_df = prod_df.sort_values("avg_rating", ascending=False)
        else:                      prod_df = prod_df.sort_values("total_orders", ascending=False)

        cols = st.columns(3)
        EMOJIS = {"Electronics":"💻","Clothing & Fashion":"👗","Home & Kitchen":"🏠",
                  "Beauty & Personal Care":"💄","Sports & Fitness":"🏋️",
                  "Books & Stationery":"📚","Toys & Games":"🎮","Groceries & Food":"🛒",
                  "Mobile Accessories":"📱","Furniture & Decor":"🛋️",
                  "Health & Wellness":"💊","Automotive":"🚗",
                  "Jewellery & Watches":"💍","Baby Products":"🍼","Pet Supplies":"🐾"}

        for i, (_, row) in enumerate(prod_df.iterrows()):
            with cols[i % 3]:
                em  = EMOJIS.get(row["category"], "📦")
                rat = "⭐" * round(row["avg_rating"])
                disc_label = f"~{row['avg_discount']:.0f}% off" if row["avg_discount"] > 0 else ""
                st.markdown(f"""
                <div style='background:#fff;border:1.5px solid #e8e4f0;border-radius:16px;
                     padding:18px;margin-bottom:14px;box-shadow:0 2px 10px rgba(0,0,0,.05)'>
                    <div style='font-size:32px;margin-bottom:8px'>{em}</div>
                    <div style='font-weight:700;font-size:15px;color:#1a1a3e;margin-bottom:4px'>
                        {row['category']}</div>
                    <div style='color:#888;font-size:13px;margin-bottom:8px'>{rat} {row['avg_rating']:.1f}</div>
                    <div style='font-size:18px;font-weight:800;color:#667eea'>
                        ₹{row['avg_price']:,.0f}</div>
                    <div style='font-size:12px;color:#27ae60;font-weight:600'>{disc_label}</div>
                    <div style='font-size:11px;color:#aaa;margin-top:4px'>
                        {row['total_orders']:,} orders</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Add to Cart", key=f"cart_{i}"):
                    st.success(f"✅ {row['category']} added to cart!")

    # ── ORDERS ────────────────────────────────────────────────────────────────
    elif "Orders" in page:
        st.markdown("## 📦 My Orders")
        status_filter = st.selectbox("Filter by Status",
                                     ["All", "delivered", "shipped", "processing", "cancelled"])
        disp = my_ord if status_filter == "All" else my_ord[my_ord["order_status"] == status_filter]

        c1, c2, c3 = st.columns(3)
        c1.metric("Showing Orders", len(disp))
        c2.metric("Total Value", fmt(disp["payment_amount"].sum()))
        c3.metric("Avg Order Value", fmt(disp["payment_amount"].mean()) if len(disp) else "₹0")

        st.dataframe(
            disp[["order_date","category","price","discount_amount","payment_amount",
                  "review_score","order_status","city"]].rename(columns={
                "order_date":"Date","category":"Category","price":"Price (₹)",
                "discount_amount":"Discount (₹)","payment_amount":"Paid (₹)",
                "review_score":"Rating","order_status":"Status","city":"City",
            }).sort_values("Date", ascending=False).reset_index(drop=True),
            use_container_width=True, height=420
        )

    # ── PROFILE ───────────────────────────────────────────────────────────────
    elif "Profile" in page:
        st.markdown("## 👤 My Profile & Behaviour Summary")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
             border-radius:18px;padding:26px 32px;margin-bottom:24px;
             display:flex;align-items:center;gap:20px'>
            <div style='width:70px;height:70px;border-radius:50%;
                 background:linear-gradient(135deg,#667eea,#f093fb);
                 display:flex;align-items:center;justify-content:center;
                 font-size:30px;font-weight:800;color:#fff;flex-shrink:0'>{name[0]}</div>
            <div>
                <div style='color:#fff;font-size:22px;font-weight:800'>{name}</div>
                <div style='color:rgba(255,255,255,.55);font-size:13px;margin-top:3px'>
                    customer@demo.com</div>
                <div style='margin-top:8px'>
                    <span style='background:rgba(102,126,234,.3);color:#a78bfa;
                        padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700'>
                        {seg_emoji} {seg}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(my_beh):
            brow = my_beh.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Orders",       int(brow["TotalOrders"]))
            c2.metric("Total Spent",        fmt(brow["TotalSpend"]))
            c3.metric("Avg Order Value",    fmt(brow["AvgOrderValue"]))
            c4.metric("Cancellation Rate",  f"{brow['CancelRate']:.1f}%")

            st.markdown("---")
            col_a, col_b = st.columns(2)
            col_a.info(f"🛍️ **Favourite Category:** {brow['FavCategory']}")
            col_a.info(f"📱 **Preferred Device:** {brow['PrefDevice']}")
            col_b.info(f"🏷️ **Avg Discount Used:** {brow['AvgDiscount']:.1f}%")
            col_b.info(f"⭐ **Avg Rating Given:** {brow['AvgRating']:.2f} / 5.0")

        st.markdown("---")
        st.subheader("📅 Spending Pattern by Day of Week")
        dow = my_ord.groupby("day_of_week")["payment_amount"].sum().reset_index()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=day_order, ordered=True)
        dow = dow.sort_values("day_of_week")
        fig = px.bar(dow, x="day_of_week", y="payment_amount",
                     color="payment_amount", color_continuous_scale=["#c7d2fe","#667eea"],
                     text_auto=True, labels={"payment_amount":"Amount (₹)","day_of_week":"Day"})
        fig.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)
