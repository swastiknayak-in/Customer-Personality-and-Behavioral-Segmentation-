import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Insights", page_icon="🔍", layout="wide")

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

df:    pd.DataFrame = st.session_state["df"]
rfm:   pd.DataFrame = st.session_state.get("rfm", pd.DataFrame())
model               = st.session_state.get("churn_model")
ft:    list         = st.session_state.get("churn_features", [])

# Friendly feature names
FEATURE_LABELS = {
    "Recency":   "Days Since Last Order",
    "Frequency": "Number of Orders",
    "Monetary":  "Total Amount Spent (₹)",
    "AvgRating": "Average Rating Given",
}

SEG_COLORS = ["#4F46E5", "#7c3aed", "#10b981", "#f59e0b"]

st.title("🔍 Customer Insights")
st.caption("Understand your customer groups and identify who might stop buying.")
st.divider()

tab1, tab2, tab3 = st.tabs(["👥 Customer Groups", "⚠️ Buying Risk Check", "📌 What Matters Most"])

# ─────────────────────────── TAB 1: CUSTOMER GROUPS ──────────────────────────
with tab1:
    st.subheader("Customer Groups")
    st.markdown(
        "We automatically group customers into **4 groups** based on "
        "how recently they bought, how often they buy, and how much they spend."
    )

    if rfm.empty:
        st.warning("Customer group data is not available right now.")
    else:
        seg_counts = rfm["Segment"].value_counts()
        cols = st.columns(len(seg_counts))
        for col, (seg, cnt) in zip(cols, seg_counts.items()):
            col.metric(seg, f"{cnt:,} customers")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            pie = seg_counts.reset_index()
            pie.columns = ["Group", "Customers"]
            fig = px.pie(
                pie, names="Group", values="Customers",
                title="Customer Group Breakdown",
                color_discrete_sequence=SEG_COLORS,
                hole=0.45,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(margin=dict(t=45, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sample = rfm.sample(min(500, len(rfm)), random_state=7)
            fig2 = px.scatter(
                sample, x="Recency", y="Monetary",
                size="Frequency", color="Segment",
                title="Customers: Days Since Order vs Total Spend",
                labels={
                    "Recency":  "Days Since Last Order",
                    "Monetary": "Total Amount Spent (₹)",
                    "Frequency":"Number of Orders",
                },
                opacity=0.7,
                color_discrete_sequence=SEG_COLORS,
            )
            fig2.update_layout(margin=dict(t=45, b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats table
        st.subheader("Group Averages")
        stats = (
            rfm.groupby("Segment")[["Recency","Frequency","Monetary"]]
            .mean().round(1).reset_index()
        )
        stats = stats.rename(columns={
            "Recency":   "Avg Days Since Order",
            "Frequency": "Avg Number of Orders",
            "Monetary":  "Avg Total Spend (₹)",
            "Segment":   "Customer Group",
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Group Comparison — Box Charts")
        c3, c4, c5 = st.columns(3)
        metrics = [
            ("Recency",   "Days Since Last Order"),
            ("Frequency", "Number of Orders"),
            ("Monetary",  "Total Spend (₹)"),
        ]
        for col, (metric, label) in zip([c3,c4,c5], metrics):
            with col:
                fig = px.box(
                    rfm, x="Segment", y=metric, color="Segment",
                    title=label,
                    labels={"Segment": "", metric: label},
                    color_discrete_sequence=SEG_COLORS,
                )
                fig.update_layout(showlegend=False, margin=dict(t=45, b=20),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────── TAB 2: BUYING RISK CHECK ────────────────────────
with tab2:
    st.subheader("Buying Risk Check")
    st.markdown(
        "Enter a customer's details to find out how likely they are to "
        "**stop buying** from your store. This helps you take action before they leave."
    )

    if model is None:
        st.warning("This feature needs more data to work. Please check back later.")
    else:
        from utils.ml_engine import predict_risk

        with st.form("risk_form"):
            c1, c2 = st.columns(2)
            recency   = c1.number_input("Days since their last order",  0,   1000, 90)
            frequency = c1.number_input("Total number of orders placed", 1,   500,  3)
            monetary  = c2.number_input("Total amount spent (₹)",        0.0, 500000.0, 2500.0, step=100.0)
            avg_rat   = c2.slider("Their average rating given", 1.0, 5.0, 4.0, 0.1)
            submitted = st.form_submit_button("Check Risk", use_container_width=True)

        if submitted:
            prob = predict_risk(model, ft, recency, frequency, monetary, avg_rat)
            pct  = prob * 100

            if prob > 0.6:
                color = "#ef4444"
                label = "🔴 High Risk"
                msg   = "This customer is very likely to stop buying. Consider sending them a special offer or discount right away."
                fn    = st.error
            elif prob > 0.35:
                color = "#f59e0b"
                label = "🟡 Medium Risk"
                msg   = "This customer may be losing interest. A personalised message or small discount could bring them back."
                fn    = st.warning
            else:
                color = "#10b981"
                label = "🟢 Low Risk"
                msg   = "This customer appears loyal and active. Keep delivering great service!"
                fn    = st.success

            st.markdown(f"### Result: **{pct:.1f}% chance of stopping** — {label}")

            gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = pct,
                number = {"suffix": "%", "font": {"size": 36}},
                gauge = {
                    "axis":  {"range": [0, 100], "tickwidth": 1},
                    "bar":   {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  35], "color": "#d1fae5"},
                        {"range": [35, 60], "color": "#fef3c7"},
                        {"range": [60, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#1e1b4b","width": 4}, "value": pct},
                },
                title = {"text": "Risk of Stopping (Churn Probability)", "font": {"size": 14}},
            ))
            gauge.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(gauge, use_container_width=True)
            fn(msg)

        # Overall risk distribution
        if not rfm.empty and "Recency" in rfm.columns:
            st.divider()
            st.subheader("Risk Levels Across All Customers")
            from utils.ml_engine import predict_risk
            rfm2 = rfm.copy()
            rfm2["Risk"] = rfm2.apply(
                lambda r: predict_risk(model, ft, int(r["Recency"]),
                                       int(r["Frequency"]), float(r["Monetary"]), 4.0),
                axis=1,
            )
            rfm2["Risk Level"] = rfm2["Risk"].apply(
                lambda v: "High Risk" if v > 0.6 else "Medium Risk" if v > 0.35 else "Low Risk"
            )
            risk_counts = rfm2["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level","Customers"]
            fig = px.bar(
                risk_counts, x="Risk Level", y="Customers",
                color="Risk Level",
                color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#10b981"},
                title="How Many Customers Are at Risk?",
                text_auto=True,
            )
            fig.update_layout(showlegend=False, margin=dict(t=45, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────── TAB 3: WHAT MATTERS MOST ────────────────────────
with tab3:
    st.subheader("What Factors Affect Customer Loyalty?")
    st.markdown(
        "This chart shows which pieces of information are most useful when "
        "predicting whether a customer will keep buying or not."
    )

    if model is None or not ft:
        st.warning("This feature needs more data to work. Please check back later.")
    else:
        imp_df = pd.DataFrame({
            "Factor":     [FEATURE_LABELS.get(f, f) for f in ft],
            "Importance": model.feature_importances_,
        }).sort_values("Importance")

        fig = px.bar(
            imp_df, x="Importance", y="Factor", orientation="h",
            title="Which Factors Predict Customer Loyalty Best?",
            labels={"Importance": "Importance Score (higher = more influential)", "Factor": ""},
            color="Importance",
            color_continuous_scale=["#c7d2fe", "#4F46E5"],
            text_auto=".2f",
        )
        fig.update_layout(margin=dict(t=45, b=20),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        if not rfm.empty:
            st.subheader("Customer Group: Spend vs Risk")
            st.caption("Each bubble is a customer group. Bigger = more revenue. Colour shows risk level.")
            from utils.ml_engine import predict_risk
            seg_agg = rfm.copy()
            seg_agg["Risk"] = seg_agg.apply(
                lambda r: predict_risk(model, ft, int(r["Recency"]),
                                       int(r["Frequency"]), float(r["Monetary"]), 4.0),
                axis=1,
            )
            agg = (
                seg_agg.groupby("Segment")
                .agg(
                    AvgSpend = ("Monetary","mean"),
                    AvgRisk  = ("Risk",    "mean"),
                    Count    = ("Monetary","count"),
                )
                .reset_index()
            )
            fig2 = px.scatter(
                agg, x="AvgRisk", y="AvgSpend",
                text="Segment", size="Count",
                color="AvgRisk",
                color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                title="Average Spend vs Average Risk by Customer Group",
                labels={
                    "AvgRisk":  "Average Risk of Stopping",
                    "AvgSpend": "Average Total Spend (₹)",
                },
            )
            fig2.update_traces(textposition="top center", textfont_size=13)
            fig2.update_layout(margin=dict(t=45, b=20),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
