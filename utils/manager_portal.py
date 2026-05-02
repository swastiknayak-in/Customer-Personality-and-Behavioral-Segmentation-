"""
Manager Portal
==============
Tabs:
  Dashboard | Customer Segmentation | Behaviour Analysis |
  Profit & Loss | Advanced Analytics | Churn Risk | Product Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PURPLE = "#667eea"
VIOLET = "#764ba2"
SEQ8   = ["#667eea","#764ba2","#f093fb","#f5576c","#fda085","#4facfe","#43e97b","#fa709a"]
SEG_COLORS = {
    "Top Customers":     "#f5576c",
    "Loyal Customers":   "#667eea",
    "Regular Customers": "#43e97b",
    "Budget Buyers":     "#fda085",
}

def fmt(n):  return f"₹{n:,.0f}"
def fmtK(n): return f"₹{n/1000:,.1f}K" if n < 1e6 else f"₹{n/1e6:,.2f}M"


def run_manager(page, df, rfm, model, feats, behavior, pnl):

    # ── DASHBOARD ─────────────────────────────────────────────────────────────
    if "Dashboard" in page:
        st.markdown("## 📊 Manager Dashboard")
        st.caption(f"Overview of all {len(df):,} orders from {df['order_date'].min().date()} to {df['order_date'].max().date()}")

        del_df = df[df["order_status"] != "cancelled"]
        total_rev  = del_df["payment_amount"].sum()
        total_ord  = len(del_df)
        avg_ord    = del_df["payment_amount"].mean()
        tot_cust   = df["customer_id"].nunique()
        avg_rat    = df["review_score"].mean()
        net_profit = pnl["net_profit"]

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Total Revenue",    fmtK(total_rev))
        c2.metric("Orders",           f"{total_ord:,}")
        c3.metric("Customers",        f"{tot_cust:,}")
        c4.metric("Avg Order Value",  fmt(avg_ord))
        c5.metric("Avg Rating",       f"{avg_rat:.2f} ⭐")
        c6.metric("Net Profit",       fmtK(net_profit), delta=f"{pnl['net_margin']:.1f}% margin")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📈 Monthly Revenue")
            monthly = (del_df.groupby("month_name")["payment_amount"]
                       .sum().reset_index()
                       .rename(columns={"payment_amount":"Revenue"}))
            fig = px.area(monthly, x="month_name", y="Revenue",
                          color_discrete_sequence=[PURPLE])
            fig.update_layout(xaxis_title="", yaxis_title="Revenue (₹)",
                              plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
                              xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("🗂 Orders by Status")
            s = df["order_status"].str.title().value_counts().reset_index()
            s.columns = ["Status","Count"]
            fig2 = px.pie(s, names="Status", values="Count",
                          color_discrete_sequence=SEQ8, hole=0.45)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("🏙️ Revenue by City")
            city = (del_df.groupby("city")["payment_amount"].sum()
                    .reset_index().sort_values("payment_amount", ascending=False)
                    .rename(columns={"payment_amount":"Revenue"}))
            fig3 = px.bar(city, x="city", y="Revenue", color="Revenue",
                          color_continuous_scale=["#c7d2fe", VIOLET],
                          text_auto=".2s")
            fig3.update_layout(xaxis_title="", yaxis_title="Revenue (₹)",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.subheader("🔥 Top 5 Categories by Revenue")
            cat = (del_df.groupby("category")["payment_amount"].sum()
                   .reset_index().sort_values("payment_amount", ascending=False).head(5)
                   .rename(columns={"payment_amount":"Revenue","category":"Category"}))
            fig4 = px.bar(cat, x="Revenue", y="Category", orientation="h",
                          color="Revenue", color_continuous_scale=["#f3e8ff", PURPLE],
                          text_auto=".2s")
            fig4.update_layout(yaxis=dict(autorange="reversed"),
                               plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📅 Order Heatmap — Day vs Hour")
        heat = df.groupby(["day_of_week","hour"]).size().reset_index(name="Orders")
        day_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat["day_of_week"] = pd.Categorical(heat["day_of_week"], categories=day_ord, ordered=True)
        pivot = heat.pivot(index="day_of_week", columns="hour", values="Orders").fillna(0)
        fig5 = px.imshow(pivot, color_continuous_scale=["#f0f0ff","#667eea"],
                         labels=dict(x="Hour of Day", y="", color="Orders"),
                         title="When do customers shop most?")
        st.plotly_chart(fig5, use_container_width=True)

    # ── CUSTOMER SEGMENTATION ─────────────────────────────────────────────────
    elif "Segmentation" in page:
        st.markdown("## 👥 Customer Segmentation — RFM Analysis")
        st.info("Customers are clustered into 4 groups using **Recency, Frequency, and Monetary** value (RFM). KMeans algorithm assigns each customer to a segment automatically.")

        seg_counts = rfm["Segment"].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        for col, (seg, cnt) in zip([c1,c2,c3,c4], seg_counts.items()):
            col.metric(seg, f"{cnt:,} customers")

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🔵 Segment Overview","📊 RFM Distributions","📋 Customer Table","🎯 Segment Profiles"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.pie(seg_counts.reset_index(), names="Segment", values="count",
                             color="Segment",
                             color_discrete_map=SEG_COLORS,
                             title="Customer Segment Distribution", hole=0.45)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                sample = rfm.sample(min(500, len(rfm)), random_state=7)
                fig2 = px.scatter(sample, x="Recency", y="Monetary",
                                  size="Frequency", color="Segment",
                                  color_discrete_map=SEG_COLORS,
                                  title="RFM Scatter — Recency vs Spend",
                                  labels={"Recency":"Days Since Last Order",
                                          "Monetary":"Total Spend (₹)",
                                          "Frequency":"Number of Orders"},
                                  opacity=0.7)
                fig2.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📊 Segment Average Statistics")
            stats = rfm.groupby("Segment")[["Recency","Frequency","Monetary","AvgRating","RFM_Score"]].mean().round(1).reset_index()
            stats.columns = ["Segment","Avg Days Since Order","Avg Orders","Avg Spend (₹)","Avg Rating","Avg RFM Score"]
            st.dataframe(stats, use_container_width=True, hide_index=True)

        with tab2:
            metric_sel = st.selectbox("Choose metric", ["Recency","Frequency","Monetary","RFM_Score"])
            label_map  = {"Recency":"Days Since Last Order","Frequency":"Number of Orders",
                          "Monetary":"Total Spend (₹)","RFM_Score":"RFM Combined Score"}
            fig3 = px.box(rfm, x="Segment", y=metric_sel, color="Segment",
                          color_discrete_map=SEG_COLORS,
                          title=f"{label_map[metric_sel]} by Customer Segment",
                          labels={"Segment":"Segment", metric_sel: label_map[metric_sel]})
            fig3.update_layout(plot_bgcolor="#fafafa", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig4 = px.histogram(rfm, x="Recency", color="Segment",
                                    color_discrete_map=SEG_COLORS,
                                    barmode="overlay", opacity=0.7,
                                    title="Recency Distribution by Segment",
                                    labels={"Recency":"Days Since Last Order"})
                fig4.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig4, use_container_width=True)
            with col_b:
                fig5 = px.histogram(rfm, x="Monetary", color="Segment",
                                    color_discrete_map=SEG_COLORS,
                                    barmode="overlay", opacity=0.7,
                                    title="Spend Distribution by Segment",
                                    labels={"Monetary":"Total Spend (₹)"})
                fig5.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig5, use_container_width=True)

        with tab3:
            seg_filter = st.multiselect("Filter by Segment", rfm["Segment"].unique().tolist(),
                                        default=rfm["Segment"].unique().tolist())
            show_rfm = rfm[rfm["Segment"].isin(seg_filter)].copy()
            show_rfm = show_rfm[["customer_id","Segment","Recency","Frequency","Monetary",
                                  "AvgRating","RFM_Score","City","Gender","AgeGroup"]].rename(
                columns={"customer_id":"Customer ID","AvgRating":"Avg Rating",
                         "RFM_Score":"RFM Score","AgeGroup":"Age Group"})
            st.dataframe(show_rfm.sort_values("RFM Score", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=420)

        with tab4:
            for seg in ["Top Customers","Loyal Customers","Regular Customers","Budget Buyers"]:
                sub = rfm[rfm["Segment"] == seg]
                if len(sub) == 0: continue
                color = SEG_COLORS[seg]
                with st.expander(f"{seg} — {len(sub):,} customers", expanded=(seg=="Top Customers")):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Count",          f"{len(sub):,}")
                    c2.metric("Avg Spend",       fmt(sub["Monetary"].mean()))
                    c3.metric("Avg Orders",      f"{sub['Frequency'].mean():.1f}")
                    c4.metric("Avg Recency",     f"{sub['Recency'].mean():.0f} days")

                    col_x, col_y = st.columns(2)
                    with col_x:
                        gc = sub["Gender"].value_counts().reset_index()
                        gc.columns = ["Gender","Count"]
                        fig = px.pie(gc, names="Gender", values="Count",
                                     color_discrete_sequence=SEQ8, hole=0.35, title="Gender Split")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_y:
                        ag = sub["AgeGroup"].value_counts().reset_index()
                        ag.columns = ["Age Group","Count"]
                        fig2 = px.bar(ag, x="Age Group", y="Count",
                                      color_discrete_sequence=[color], text_auto=True,
                                      title="Age Group Distribution")
                        fig2.update_layout(plot_bgcolor="#fafafa", showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)

    # ── BEHAVIOUR ANALYSIS ────────────────────────────────────────────────────
    elif "Behaviour" in page:
        st.markdown("## 🧠 Customer Behaviour Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(["🕒 Time Patterns","📱 Device & Demographics","🛍️ Category Behaviour","⭐ Ratings & Discounts"])

        with tab1:
            st.subheader("Shopping Hours Distribution")
            hour_data = df.groupby("hour").size().reset_index(name="Orders")
            fig = px.bar(hour_data, x="hour", y="Orders",
                         color="Orders", color_continuous_scale=["#f0f0ff","#667eea"],
                         text_auto=True, title="Orders by Hour of Day",
                         labels={"hour":"Hour (24h)","Orders":"Number of Orders"})
            fig.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                dow = df.groupby("day_of_week").size().reset_index(name="Orders")
                day_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=day_ord, ordered=True)
                fig2 = px.bar(dow.sort_values("day_of_week"), x="day_of_week", y="Orders",
                              color="Orders", color_continuous_scale=["#f0f0ff","#764ba2"],
                              title="Orders by Day of Week", text_auto=True)
                fig2.update_layout(xaxis_title="", plot_bgcolor="#fafafa", paper_bgcolor="#fafafa")
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                month = df.groupby("month_name").size().reset_index(name="Orders")
                fig3 = px.line(month, x="month_name", y="Orders", markers=True,
                               color_discrete_sequence=[PURPLE],
                               title="Orders by Month",
                               labels={"month_name":"Month"})
                fig3.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
                                   xaxis_tickangle=-40)
                st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                dev = df.groupby("device").agg(Orders=("order_date","count"),
                                               Revenue=("payment_amount","sum")).reset_index()
                fig = px.bar(dev, x="device", y="Orders", color="device",
                             color_discrete_sequence=SEQ8, text_auto=True,
                             title="Orders by Device")
                fig.update_layout(showlegend=False, plot_bgcolor="#fafafa")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                gen = df.groupby("gender").agg(Orders=("order_date","count"),
                                               Revenue=("payment_amount","sum")).reset_index()
                fig2 = px.pie(gen, names="gender", values="Orders",
                              color_discrete_sequence=SEQ8, hole=0.4,
                              title="Orders by Gender")
                st.plotly_chart(fig2, use_container_width=True)

            age = df.groupby("age_group").agg(
                Orders  = ("order_date","count"),
                Revenue = ("payment_amount","sum"),
                AvgSpend= ("payment_amount","mean")).reset_index()
            fig3 = px.bar(age, x="age_group", y="Revenue", color="age_group",
                          color_discrete_sequence=SEQ8, text_auto=".2s",
                          title="Revenue by Age Group",
                          labels={"age_group":"Age Group","Revenue":"Revenue (₹)"})
            fig3.update_layout(showlegend=False, plot_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("📊 Avg Order Value by Gender × Age Group")
            pivot_aov = df.groupby(["gender","age_group"])["payment_amount"].mean().reset_index()
            fig4 = px.density_heatmap(pivot_aov, x="age_group", y="gender", z="payment_amount",
                                      color_continuous_scale=["#f0f0ff","#667eea"],
                                      title="Average Order Value Heatmap",
                                      labels={"payment_amount":"Avg Order Value (₹)",
                                              "age_group":"Age Group","gender":"Gender"})
            st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            cat_beh = df.groupby("category").agg(
                Orders      = ("order_date","count"),
                Revenue     = ("payment_amount","sum"),
                AvgSpend    = ("payment_amount","mean"),
                AvgRating   = ("review_score","mean"),
                AvgDiscount = ("discount_pct","mean"),
                CancelRate  = ("order_status", lambda x: (x=="cancelled").mean()*100),
            ).reset_index().sort_values("Revenue", ascending=False)

            st.dataframe(cat_beh.rename(columns={
                "category":"Category","Orders":"Total Orders","Revenue":"Revenue (₹)",
                "AvgSpend":"Avg Order (₹)","AvgRating":"Avg Rating",
                "AvgDiscount":"Avg Discount %","CancelRate":"Cancel Rate %"
            }).round(1).reset_index(drop=True), use_container_width=True, height=420)

            col_a, col_b = st.columns(2)
            with col_a:
                fig5 = px.scatter(cat_beh, x="AvgSpend", y="AvgRating",
                                  size="Orders", color="Revenue",
                                  hover_name="category",
                                  color_continuous_scale=["#f0f0ff","#764ba2"],
                                  title="Avg Spend vs Rating (bubble = orders)",
                                  labels={"AvgSpend":"Avg Spend (₹)","AvgRating":"Avg Rating"})
                fig5.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig5, use_container_width=True)

            with col_b:
                fig6 = px.bar(cat_beh.sort_values("CancelRate", ascending=False),
                              x="CancelRate", y="category", orientation="h",
                              color="CancelRate",
                              color_continuous_scale=["#ecfdf5","#f5576c"],
                              title="Cancellation Rate by Category (%)",
                              labels={"CancelRate":"Cancel Rate (%)","category":"Category"})
                fig6.update_layout(plot_bgcolor="#fafafa", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig6, use_container_width=True)

        with tab4:
            col_a, col_b = st.columns(2)
            with col_a:
                rat_dist = df["review_score"].value_counts().sort_index().reset_index()
                rat_dist.columns = ["Rating","Count"]
                fig7 = px.bar(rat_dist, x="Rating", y="Count",
                              color="Rating",
                              color_continuous_scale=["#ef4444","#f59e0b","#667eea","#22c55e","#15803d"],
                              text_auto=True, title="Overall Rating Distribution")
                fig7.update_layout(plot_bgcolor="#fafafa", showlegend=False)
                st.plotly_chart(fig7, use_container_width=True)

            with col_b:
                disc_seg = df[df["discount_pct"] > 0].groupby("age_group")["discount_pct"].mean().reset_index()
                fig8 = px.bar(disc_seg, x="age_group", y="discount_pct",
                              color="age_group", color_discrete_sequence=SEQ8,
                              text_auto=".1f", title="Avg Discount % by Age Group",
                              labels={"age_group":"Age Group","discount_pct":"Avg Discount %"})
                fig8.update_layout(showlegend=False, plot_bgcolor="#fafafa")
                st.plotly_chart(fig8, use_container_width=True)

            st.subheader("⭐ Avg Rating by Category")
            rat_cat = df.groupby("category")["review_score"].mean().sort_values(ascending=False).reset_index()
            fig9 = px.bar(rat_cat, x="review_score", y="category", orientation="h",
                          color="review_score",
                          color_continuous_scale=["#fecaca","#f59e0b","#22c55e"],
                          labels={"review_score":"Avg Rating","category":"Category"},
                          text_auto=".2f")
            fig9.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="#fafafa")
            st.plotly_chart(fig9, use_container_width=True)

    # ── PROFIT & LOSS ─────────────────────────────────────────────────────────
    elif "Profit" in page:
        st.markdown("## 💰 Profit & Loss Statement")

        # Summary KPIs
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Revenue",   fmtK(pnl["revenue"]))
        c2.metric("Total COGS",      fmtK(pnl["cogs"]))
        c3.metric("Gross Profit",    fmtK(pnl["gross_profit"]),
                  delta=f"{pnl['gross_margin']:.1f}% margin")
        c4.metric("Net Profit",      fmtK(pnl["net_profit"]),
                  delta=f"{pnl['net_margin']:.1f}% margin")
        c5.metric("Discount Given",  fmtK(pnl["discount_cost"]))

        st.markdown("---")

        # Waterfall P&L
        st.subheader("📊 P&L Waterfall Chart")
        wf_labels = ["Revenue","COGS","Delivery Cost","Discount Cost","Gross Profit","Marketing","Ops Overhead","Net Profit"]
        wf_values = [
            pnl["revenue"], -pnl["cogs"], -pnl["delivery_cost"],
            -pnl["discount_cost"], None, -pnl["marketing"],
            -pnl["ops_overhead"], None
        ]
        measures = ["relative","relative","relative","relative","total","relative","relative","total"]
        fig_wf = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=measures,
            x=wf_labels,
            y=[pnl["revenue"], -pnl["cogs"], -pnl["delivery_cost"],
               -pnl["discount_cost"], pnl["gross_profit"], -pnl["marketing"],
               -pnl["ops_overhead"], pnl["net_profit"]],
            connector={"line":{"color":"rgba(63,63,63,0.3)"}},
            increasing={"marker":{"color":"#22c55e"}},
            decreasing={"marker":{"color":"#ef4444"}},
            totals={"marker":{"color":"#667eea"}},
        ))
        fig_wf.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
                             yaxis_title="Amount (₹)", height=400)
        st.plotly_chart(fig_wf, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📅 Monthly Revenue vs Net Profit")
            monthly = pnl["monthly"].copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Revenue", x=monthly["month_name"],
                                  y=monthly["Revenue"], marker_color="#667eea"))
            fig2.add_trace(go.Bar(name="Gross Profit", x=monthly["month_name"],
                                  y=monthly["GrossProfit"], marker_color="#43e97b"))
            fig2.add_trace(go.Scatter(name="Net Profit", x=monthly["month_name"],
                                      y=monthly["NetProfit"], mode="lines+markers",
                                      line=dict(color="#f5576c", width=2.5)))
            fig2.update_layout(barmode="group", plot_bgcolor="#fafafa",
                               paper_bgcolor="#fafafa", xaxis_tickangle=-40,
                               legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("🗂 Profit by Category")
            cat_pnl = pnl["cat_pnl"]
            fig3 = px.bar(cat_pnl, x="GrossProfit", y="category", orientation="h",
                          color="Margin%",
                          color_continuous_scale=["#fee2e2","#bbf7d0","#16a34a"],
                          labels={"GrossProfit":"Gross Profit (₹)","category":"Category",
                                  "Margin%":"Margin %"},
                          text_auto=".2s")
            fig3.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📋 Detailed P&L by Category")
        st.dataframe(
            cat_pnl.rename(columns={"category":"Category","Revenue":"Revenue (₹)",
                                    "COGS":"COGS (₹)","GrossProfit":"Gross Profit (₹)",
                                    "Orders":"Total Orders","Margin%":"Margin %"})
            .round(0).reset_index(drop=True),
            use_container_width=True, height=400
        )

        st.subheader("💸 Cost Breakdown")
        cost_labels = ["COGS","Delivery Cost","Discount Losses","Marketing","Ops Overhead"]
        cost_vals   = [pnl["cogs"], pnl["delivery_cost"], pnl["discount_cost"],
                       pnl["marketing"], pnl["ops_overhead"]]
        fig4 = px.pie(values=cost_vals, names=cost_labels,
                      color_discrete_sequence=["#ef4444","#f59e0b","#667eea","#a78bfa","#22c55e"],
                      hole=0.45, title="Cost Component Breakdown")
        st.plotly_chart(fig4, use_container_width=True)

    # ── ADVANCED ANALYTICS ────────────────────────────────────────────────────
    elif "Advanced" in page:
        st.markdown("## 📈 Advanced Analytics")

        tab1, tab2, tab3 = st.tabs(["🔁 Repeat & Retention","📊 Cohort Summary","🌍 Geo Analysis"])

        with tab1:
            ord_per_cust = df.groupby("customer_id").size().reset_index(name="Orders")
            c1,c2,c3 = st.columns(3)
            repeat_rate = (ord_per_cust["Orders"] > 1).mean() * 100
            heavy       = (ord_per_cust["Orders"] >= 5).mean() * 100
            one_time    = (ord_per_cust["Orders"] == 1).mean() * 100
            c1.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
            c2.metric("Heavy Buyers (5+ orders)", f"{heavy:.1f}%")
            c3.metric("One-Time Buyers", f"{one_time:.1f}%")

            fig = px.histogram(ord_per_cust, x="Orders", nbins=20,
                               color_discrete_sequence=[PURPLE],
                               title="Distribution of Orders per Customer",
                               labels={"Orders":"Number of Orders"})
            fig.update_layout(plot_bgcolor="#fafafa")
            st.plotly_chart(fig, use_container_width=True)

            top10 = (df.groupby("customer_id")["payment_amount"].sum()
                     .sort_values(ascending=False).head(10).reset_index()
                     .rename(columns={"customer_id":"Customer","payment_amount":"Total Spend (₹)"}))
            st.subheader("🏆 Top 10 Customers by Spend")
            st.dataframe(top10, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("📅 Monthly New vs Returning Customers")
            first_order = df.groupby("customer_id")["order_date"].min().reset_index()
            first_order["first_month"] = first_order["order_date"].dt.to_period("M").astype(str)
            df2 = df.merge(first_order[["customer_id","first_month"]], on="customer_id")
            df2["order_month"] = df2["order_date"].dt.to_period("M").astype(str)
            df2["cust_type"] = np.where(df2["order_month"] == df2["first_month"], "New", "Returning")
            cohort = df2.groupby(["order_month","cust_type"]).size().reset_index(name="Customers")
            fig2 = px.bar(cohort, x="order_month", y="Customers", color="cust_type",
                          color_discrete_sequence=["#667eea","#43e97b"],
                          barmode="stack", title="New vs Returning Customers per Month",
                          labels={"order_month":"Month","cust_type":"Customer Type"})
            fig2.update_layout(xaxis_tickangle=-40, plot_bgcolor="#fafafa")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📊 Revenue: New vs Returning")
            rev_cohort = df2.groupby(["order_month","cust_type"])["payment_amount"].sum().reset_index()
            fig3 = px.bar(rev_cohort, x="order_month", y="payment_amount", color="cust_type",
                          color_discrete_sequence=["#667eea","#43e97b"],
                          barmode="stack", title="Revenue from New vs Returning Customers",
                          labels={"order_month":"Month","payment_amount":"Revenue (₹)","cust_type":"Type"})
            fig3.update_layout(xaxis_tickangle=-40, plot_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            city_full = df.groupby("city").agg(
                Revenue  = ("payment_amount","sum"),
                Orders   = ("order_date","count"),
                Customers= ("customer_id","nunique"),
                AvgOrder = ("payment_amount","mean"),
            ).reset_index().sort_values("Revenue", ascending=False)

            st.dataframe(city_full.rename(columns={
                "city":"City","Revenue":"Revenue (₹)","Orders":"Orders",
                "Customers":"Unique Customers","AvgOrder":"Avg Order (₹)"
            }).round(0).reset_index(drop=True), use_container_width=True)

            fig4 = px.treemap(city_full, path=["city"], values="Revenue",
                              color="AvgOrder",
                              color_continuous_scale=["#e0e7ff","#667eea"],
                              title="Revenue Treemap by City",
                              labels={"AvgOrder":"Avg Order (₹)"})
            st.plotly_chart(fig4, use_container_width=True)

    # ── CHURN RISK ────────────────────────────────────────────────────────────
    elif "Churn" in page:
        st.markdown("## ⚠️ Customer Churn Risk Analysis")
        st.info("A customer is considered **at risk** if they have not ordered in the last 180 days. The model uses Recency, Frequency, Monetary value, and Avg Rating to predict risk probability.")

        if model is None:
            st.error("Model could not be trained (insufficient data).")
            return

        rfm2 = rfm.copy()
        rfm2["ChurnProb"] = rfm2.apply(
            lambda r: _pred(model, feats, int(r["Recency"]),
                            int(r["Frequency"]), float(r["Monetary"]), float(r["AvgRating"])),
            axis=1
        )
        rfm2["Risk Level"] = rfm2["ChurnProb"].apply(
            lambda p: "High Risk" if p > 0.6 else "Medium Risk" if p > 0.35 else "Low Risk"
        )

        c1,c2,c3 = st.columns(3)
        hr = (rfm2["Risk Level"] == "High Risk").sum()
        mr = (rfm2["Risk Level"] == "Medium Risk").sum()
        lr = (rfm2["Risk Level"] == "Low Risk").sum()
        c1.metric("🔴 High Risk Customers",   f"{hr:,}", delta=f"{hr/len(rfm2)*100:.1f}%")
        c2.metric("🟡 Medium Risk Customers", f"{mr:,}", delta=f"{mr/len(rfm2)*100:.1f}%")
        c3.metric("🟢 Low Risk Customers",    f"{lr:,}", delta=f"{lr/len(rfm2)*100:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            rc = rfm2["Risk Level"].value_counts().reset_index()
            rc.columns = ["Risk Level","Count"]
            fig = px.pie(rc, names="Risk Level", values="Count",
                         color="Risk Level",
                         color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
                         hole=0.45, title="Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.histogram(rfm2, x="ChurnProb", nbins=30,
                                color_discrete_sequence=[PURPLE],
                                title="Churn Probability Distribution",
                                labels={"ChurnProb":"Churn Probability (0 = safe, 1 = at risk)"})
            fig2.update_layout(plot_bgcolor="#fafafa")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔬 Feature Importance — What drives churn?")
        feat_labels = {"Recency":"Days Since Last Order","Frequency":"Number of Orders",
                       "Monetary":"Total Spend (₹)","AvgRating":"Average Rating Given"}
        fi_df = pd.DataFrame({
            "Factor":     [feat_labels.get(f, f) for f in feats],
            "Importance": model.feature_importances_,
        }).sort_values("Importance")
        fig3 = px.bar(fi_df, x="Importance", y="Factor", orientation="h",
                      color="Importance", color_continuous_scale=["#e0e7ff","#667eea"],
                      title="Which factors best predict churn?",
                      text_auto=".3f")
        fig3.update_layout(plot_bgcolor="#fafafa")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🎯 Interactive Churn Predictor")
        st.markdown("Enter a customer's details to see their churn risk:")
        col_i1, col_i2 = st.columns(2)
        rec = col_i1.number_input("Days since last order",     0, 1000, 90)
        frq = col_i1.number_input("Total orders placed",       1, 500,  3)
        mon = col_i2.number_input("Total amount spent (₹)",    0.0, 500000.0, 2500.0, step=100.0)
        rat = col_i2.slider("Average rating they give",        1.0, 5.0, 4.0, 0.1)

        if st.button("Calculate Churn Risk", use_container_width=True):
            prob = _pred(model, feats, rec, frq, mon, rat)
            pct  = prob * 100
            if prob > 0.6:
                st.error(f"🔴 **High Risk — {pct:.1f}% churn probability.** This customer is likely to leave. Send a re-engagement offer immediately.")
            elif prob > 0.35:
                st.warning(f"🟡 **Medium Risk — {pct:.1f}% churn probability.** Monitor this customer and consider a personalised discount.")
            else:
                st.success(f"🟢 **Low Risk — {pct:.1f}% churn probability.** Customer appears loyal and active. Keep the good service going!")
            _gauge(pct)

        st.markdown("---")
        st.subheader("📋 High Risk Customers — Action Required")
        high_risk = rfm2[rfm2["Risk Level"] == "High Risk"][
            ["customer_id","Segment","Recency","Frequency","Monetary","AvgRating","ChurnProb","City"]
        ].sort_values("ChurnProb", ascending=False).reset_index(drop=True)
        high_risk["ChurnProb"] = (high_risk["ChurnProb"] * 100).round(1).astype(str) + "%"
        high_risk.columns = ["Customer ID","Segment","Days Inactive","Orders","Total Spend (₹)",
                              "Avg Rating","Churn Risk","City"]
        st.dataframe(high_risk, use_container_width=True, height=380)

    # ── PRODUCT ANALYTICS ─────────────────────────────────────────────────────
    elif "Product" in page:
        st.markdown("## 🛒 Product & Category Analytics")

        tab1, tab2 = st.tabs(["📦 Category Deep Dive","💡 Insights"])

        with tab1:
            cat_stats = df.groupby("category").agg(
                Orders      = ("order_date","count"),
                Revenue     = ("payment_amount","sum"),
                AvgPrice    = ("price","mean"),
                AvgDisc     = ("discount_pct","mean"),
                AvgRating   = ("review_score","mean"),
                CancelRate  = ("order_status", lambda x: (x=="cancelled").mean()*100),
                UniqueCustomers = ("customer_id","nunique"),
            ).reset_index().sort_values("Revenue", ascending=False)

            sel = st.selectbox("Drill into a category", cat_stats["category"].tolist())
            sub = df[df["category"] == sel]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Orders",  f"{len(sub):,}")
            c2.metric("Revenue",       fmtK(sub["payment_amount"].sum()))
            c3.metric("Avg Price",     fmt(sub["price"].mean()))
            c4.metric("Avg Rating",    f"{sub['review_score'].mean():.2f} ⭐")

            col_a, col_b = st.columns(2)
            with col_a:
                m_sub = sub.groupby("month_name")["payment_amount"].sum().reset_index()
                fig = px.line(m_sub, x="month_name", y="payment_amount", markers=True,
                              color_discrete_sequence=[PURPLE],
                              title=f"Monthly Revenue — {sel}",
                              labels={"month_name":"Month","payment_amount":"Revenue (₹)"})
                fig.update_layout(plot_bgcolor="#fafafa", xaxis_tickangle=-40)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                cit_sub = sub.groupby("city")["payment_amount"].sum().reset_index().sort_values("payment_amount", ascending=False)
                fig2 = px.bar(cit_sub, x="city", y="payment_amount",
                              color="payment_amount", color_continuous_scale=["#f0f0ff","#764ba2"],
                              title=f"Revenue by City — {sel}",
                              labels={"city":"City","payment_amount":"Revenue (₹)"},
                              text_auto=".2s")
                fig2.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("🏆 Category Performance Matrix")
            fig3 = px.scatter(cat_stats, x="AvgDisc", y="AvgRating",
                              size="Revenue", color="category",
                              hover_name="category",
                              color_discrete_sequence=SEQ8,
                              title="Avg Discount % vs Avg Rating (bubble = revenue)",
                              labels={"AvgDisc":"Avg Discount %","AvgRating":"Avg Rating"})
            fig3.update_layout(plot_bgcolor="#fafafa")
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("📉 Categories with Highest Cancellation Rates")
            fig4 = px.bar(cat_stats.sort_values("CancelRate", ascending=False),
                          x="CancelRate", y="category", orientation="h",
                          color="CancelRate",
                          color_continuous_scale=["#d1fae5","#fef3c7","#fee2e2"],
                          title="Cancellation Rate by Category (%)",
                          text_auto=".1f",
                          labels={"CancelRate":"Cancel Rate (%)","category":"Category"})
            fig4.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="#fafafa")
            st.plotly_chart(fig4, use_container_width=True)


def _pred(model, feats, recency, frequency, monetary, avg_rating):
    import pandas as pd
    X = pd.DataFrame([[recency, frequency, monetary, avg_rating]], columns=feats)
    return float(model.predict_proba(X)[0][1])


def _gauge(pct):
    import plotly.graph_objects as go
    color = "#ef4444" if pct > 60 else "#f59e0b" if pct > 35 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix":"%","font":{"size":36}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":color,"thickness":0.28},
            "steps":[
                {"range":[0,35],"color":"#d1fae5"},
                {"range":[35,60],"color":"#fef3c7"},
                {"range":[60,100],"color":"#fee2e2"},
            ],
            "threshold":{"line":{"color":"#1a1a3e","width":3},"value":pct},
        },
        title={"text":"Churn Risk Level","font":{"size":14}},
    ))
    fig.update_layout(height=260, margin=dict(t=50,b=10,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)
