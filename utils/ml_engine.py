import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and enriches the raw order dataframe.
    Accepts both real Olist CSVs and mock_data output.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"], errors="coerce"
        )

    # Fill numeric nulls
    for col in ["price", "freight_value", "payment_value", "review_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls
    if "product_category_name" in df.columns:
        df["product_category_name"] = (
            df["product_category_name"].fillna("unknown").astype(str)
        )
    if "order_status" in df.columns:
        df["order_status"] = df["order_status"].fillna("unknown").astype(str)

    # Derived columns
    if "order_purchase_timestamp" in df.columns:
        df["year"]  = df["order_purchase_timestamp"].dt.year
        df["month"] = df["order_purchase_timestamp"].dt.month
        df["dow"]   = df["order_purchase_timestamp"].dt.dayofweek

    return df


# ────────────────────────────────────────────────────────────────────────────
# RFM Segmentation
# ────────────────────────────────────────────────────────────────────────────

def train_rfm_segments(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Compute RFM features per customer and cluster them with KMeans.
    Returns a per-customer dataframe with segment labels.
    """
    if "customer_unique_id" not in df.columns or df.empty:
        return pd.DataFrame()

    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            Recency   = ("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
            Frequency = ("order_purchase_timestamp", "count"),
            Monetary  = ("payment_value", "sum"),
        )
        .reset_index()
    )

    # Scale
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    # Map clusters to human-readable labels based on mean Monetary per cluster
    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values()
    label_map = {}
    labels = ["Low Value", "Medium Value", "High Value", "Champions"]
    for rank, cluster_id in enumerate(cluster_means.index):
        label_map[cluster_id] = labels[min(rank, len(labels) - 1)]

    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm


# ────────────────────────────────────────────────────────────────────────────
# Churn Model
# ────────────────────────────────────────────────────────────────────────────

def train_churn_model(df: pd.DataFrame):
    """
    Trains a simple churn classifier.
    'Churned' = customer last purchased > 180 days ago.
    Returns (model, feature_names) or (None, []) if insufficient data.
    """
    if "customer_unique_id" not in df.columns or df.empty:
        return None, []

    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    cust = (
        df.groupby("customer_unique_id")
        .agg(
            Recency   = ("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
            Frequency = ("order_purchase_timestamp", "count"),
            Monetary  = ("payment_value", "sum"),
            AvgReview = ("review_score", "mean"),
        )
        .reset_index()
    )

    cust["Churned"] = (cust["Recency"] > 180).astype(int)

    features = ["Recency", "Frequency", "Monetary", "AvgReview"]
    X = cust[features].fillna(0)
    y = cust["Churned"]

    if len(y.unique()) < 2 or len(X) < 20:
        return None, []

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, features


def predict_churn_proba(model, features: list, recency: int,
                        frequency: int, monetary: float, avg_review: float) -> float:
    """Return the churn probability (0-1) for a single customer."""
    if model is None:
        return 0.5
    X = pd.DataFrame([[recency, frequency, monetary, avg_review]], columns=features)
    return float(model.predict_proba(X)[0][1])
