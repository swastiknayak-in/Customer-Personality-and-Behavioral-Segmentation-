import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML Insights · Olist", page_icon="🤖", layout="wide")

if not st.session_state.get("logged_in"):
    st.warning("Please log in from the Home page.")
    st.stop()

df:    pd.DataFrame = st.session_state["df"]
rfm:   pd.DataFrame = st.session_state.get("rfm", pd.DataFrame())
model               = st.session_state.get("churn_model")
ft:    list         = st.session_state.get("churn_features", [])

st.title("🤖 ML Insights")
st.caption("RFM Customer Segmentation · Churn Prediction · Feature Importance")
st.divider()

tab1, tab2, tab3 = st.tabs(["📍 RFM Segments", "⚠️ Churn Predictor", "🔬 Feature Importance"])

# ─────────────────────────── TAB 1: RFM ───────────────────────────────────────
with tab1:
    st.subheader("Customer Segmentation via RFM + KMeans")
    st.markdown(
        "Each customer is scored on **Recency**, **Frequency**, and **Monetary** value "
        "and clustered into four segments using KMeans."
    )

    if rfm.empty:
        st.warning("RFM data not available.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        seg_counts = rfm["Segment"].value_counts()
        for col, seg in zip([k1, k2, k3, k4], seg_counts.index):
            col.metric(seg, f"{seg_counts[seg]:,} customers")

        c1, c2 = st.columns(2)

        with c1:
            pie = rfm["Segment"].value_counts().reset_index()
            pie.columns = ["Segment", "Count"]
            fig = px.pie(pie, names="Segment", values="Count",
                         title="Segment Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.scatter(rfm.sample(min(500, len(rfm)), random_state=7),
                              x="Recency", y="Monetary",
                              size="Frequency", color="Segment",
                              title="RFM Scatter (sample 500)",
                              opacity=0.7,
                              color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Segment Statistics")
        stats = (
            rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .round(1)
            .reset_index()
        )
        st.dataframe(stats, use_container_width=True)

        # Box plots
        c3, c4, c5 = st.columns(3)
        for col, metric in zip([c3, c4, c5], ["Recency", "Frequency", "Monetary"]):
            with col:
                fig = px.box(rfm, x="Segment", y=metric, color="Segment",
                             title=f"{metric} by Segment",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────── TAB 2: CHURN ─────────────────────────────────────
with tab2:
    st.subheader("⚠️ Churn Probability Estimator")
    st.markdown(
        "Enter a customer's metrics below and get an instant churn probability "
        "from the trained **Random Forest** model."
    )

    if model is None:
        st.warning("Churn model not available (insufficient training data).")
    else:
        from utils.ml_engine import predict_churn_proba

        with st.form("churn_form"):
            col1, col2 = st.columns(2)
            recency   = col1.number_input("Days since last purchase (Recency)", 0, 1000, 90)
            frequency = col1.number_input("Number of orders (Frequency)", 1, 200, 3)
            monetary  = col2.number_input("Total spend R$ (Monetary)", 0.0, 50000.0, 250.0, step=10.0)
            avg_rev   = col2.slider("Average Review Score", 1.0, 5.0, 4.0, 0.1)
            submit    = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

        if submit:
            prob = predict_churn_proba(model, ft, recency, frequency, monetary, avg_rev)
            pct  = prob * 100

            color = "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.35 else "#27ae60"
            label = "🔴 High Risk" if prob > 0.6 else "🟡 Medium Risk" if prob > 0.35 else "🟢 Low Risk"

            st.markdown(f"### Churn Probability: **{pct:.1f}%** — {label}")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  35], "color": "#d5f5e3"},
                        {"range": [35, 60], "color": "#fdebd0"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": pct},
                },
                title={"text": "Churn Risk Gauge"},
            ))
            gauge.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            if prob > 0.6:
                st.error("🚨 This customer is at high risk of churning. Consider a retention offer.")
            elif prob > 0.35:
                st.warning("⚠️ Moderate churn risk. A personalised discount could help.")
            else:
                st.success("✅ This customer appears loyal. Keep up the good service!")

        # Distribution of churn risk across the full customer base
        if not rfm.empty and "Recency" in rfm.columns:
            st.divider()
            st.subheader("Churn Risk Distribution Across All Customers")
            rfm2 = rfm.copy()
            rfm2["ChurnProb"] = rfm2.apply(
                lambda r: predict_churn_proba(model, ft,
                                              int(r["Recency"]), int(r["Frequency"]),
                                              float(r["Monetary"]), 4.0),
                axis=1,
            )
            fig = px.histogram(rfm2, x="ChurnProb", nbins=30,
                               title="Churn Probability Distribution",
                               labels={"ChurnProb": "Churn Probability"},
                               color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────── TAB 3: FEATURE IMPORTANCE ───────────────────────
with tab3:
    st.subheader("🔬 Feature Importance")
    st.markdown("Which features drive the churn model's decisions?")

    if model is None or not ft:
        st.warning("Model not available.")
    else:
        imp_df = pd.DataFrame({
            "Feature":   ft,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     title="Random Forest Feature Importances",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📊 Revenue vs Churn Risk by Segment")
        if not rfm.empty:
            from utils.ml_engine import predict_churn_proba
            seg_stats = rfm.copy()
            seg_stats["ChurnProb"] = seg_stats.apply(
                lambda r: predict_churn_proba(model, ft,
                                              int(r["Recency"]), int(r["Frequency"]),
                                              float(r["Monetary"]), 4.0),
                axis=1,
            )
            agg = (
                seg_stats.groupby("Segment")
                .agg(AvgRevenue=("Monetary", "mean"), AvgChurnRisk=("ChurnProb", "mean"))
                .reset_index()
            )
            fig2 = px.scatter(agg, x="AvgChurnRisk", y="AvgRevenue",
                              text="Segment", size="AvgRevenue",
                              color="AvgChurnRisk", color_continuous_scale="RdYlGn_r",
                              title="Avg Revenue vs Avg Churn Risk by Segment")
            fig2.update_traces(textposition="top center")
            fig2.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
