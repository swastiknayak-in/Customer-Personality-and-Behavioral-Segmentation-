import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

INR = lambda v: f"₹{v:,.0f}"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and enriches raw order data."""
    df = df.copy()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    for col in ["price", "delivery_charge", "payment_amount", "review_score", "discount_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in ["category", "order_status", "city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    if "order_date" in df.columns:
        df["year"]  = df["order_date"].dt.year
        df["month"] = df["order_date"].dt.month
        df["dow"]   = df["order_date"].dt.dayofweek

    return df


def train_rfm_segments(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Clusters customers into value segments based on:
      Recency   – how recently they bought
      Frequency – how often they buy
      Monetary  – how much they spend
    """
    if "customer_id" not in df.columns or df.empty:
        return pd.DataFrame()

    snapshot = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency   = ("order_date",      lambda x: (snapshot - x.max()).days),
            Frequency = ("order_date",      "count"),
            Monetary  = ("payment_amount",  "sum"),
        )
        .reset_index()
    )

    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values()
    labels        = ["Budget Buyers", "Regular Customers", "Loyal Customers", "Top Customers"]
    label_map     = {cid: labels[min(i, 3)] for i, cid in enumerate(cluster_means.index)}
    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm


def train_churn_model(df: pd.DataFrame):
    """
    Trains a model to predict which customers may stop buying.
    A customer is considered at risk if they haven't ordered in 180+ days.
    Returns (model, feature_names).
    """
    if "customer_id" not in df.columns or df.empty:
        return None, []

    snapshot = df["order_date"].max() + pd.Timedelta(days=1)

    cust = (
        df.groupby("customer_id")
        .agg(
            Recency   = ("order_date",     lambda x: (snapshot - x.max()).days),
            Frequency = ("order_date",     "count"),
            Monetary  = ("payment_amount", "sum"),
            AvgRating = ("review_score",   "mean"),
        )
        .reset_index()
    )

    cust["AtRisk"] = (cust["Recency"] > 180).astype(int)
    features = ["Recency", "Frequency", "Monetary", "AvgRating"]
    X = cust[features].fillna(0)
    y = cust["AtRisk"]

    if len(y.unique()) < 2 or len(X) < 20:
        return None, []

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, features


def predict_risk(model, features: list,
                 recency: int, frequency: int,
                 monetary: float, avg_rating: float) -> float:
    """Returns probability (0–1) that a customer will stop buying."""
    if model is None:
        return 0.5
    X = pd.DataFrame([[recency, frequency, monetary, avg_rating]], columns=features)
    return float(model.predict_proba(X)[0][1])
