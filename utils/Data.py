"""
utils/data.py
Generates synthetic Indian e-commerce data and runs all ML models.
Everything is computed once and cached via st.cache_data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics", "Clothing & Fashion", "Home & Kitchen",
    "Beauty & Personal Care", "Sports & Fitness", "Books & Stationery",
    "Toys & Games", "Groceries & Food", "Mobile Accessories",
    "Furniture & Decor", "Health & Wellness", "Automotive",
    "Jewellery & Watches", "Baby Products", "Pet Supplies",
]

PRICE_BANDS = {
    "Electronics":            (999,  49999),
    "Clothing & Fashion":     (299,   4999),
    "Home & Kitchen":         (199,   9999),
    "Beauty & Personal Care": (99,    2999),
    "Sports & Fitness":       (499,  14999),
    "Books & Stationery":     (99,    1499),
    "Toys & Games":           (199,   4999),
    "Groceries & Food":       (49,    1999),
    "Mobile Accessories":     (199,   4999),
    "Furniture & Decor":      (999,  29999),
    "Health & Wellness":      (149,   3999),
    "Automotive":             (299,   9999),
    "Jewellery & Watches":    (499,  24999),
    "Baby Products":          (299,   5999),
    "Pet Supplies":           (199,   3999),
}

COST_RATIO = {          # cost as fraction of selling price (for P&L)
    "Electronics":            0.62,
    "Clothing & Fashion":     0.40,
    "Home & Kitchen":         0.52,
    "Beauty & Personal Care": 0.38,
    "Sports & Fitness":       0.48,
    "Books & Stationery":     0.35,
    "Toys & Games":           0.44,
    "Groceries & Food":       0.70,
    "Mobile Accessories":     0.55,
    "Furniture & Decor":      0.50,
    "Health & Wellness":      0.42,
    "Automotive":             0.58,
    "Jewellery & Watches":    0.45,
    "Baby Products":          0.46,
    "Pet Supplies":           0.47,
}

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]

GENDERS    = ["Male", "Female", "Other"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
DEVICES    = ["Mobile", "Desktop", "Tablet"]
STATUSES   = ["delivered", "shipped", "processing", "cancelled"]
STATUS_W   = [0.78, 0.11, 0.08, 0.03]


# ── Data generation ───────────────────────────────────────────────────────────
def generate_data(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    n_customers  = max(80, n_rows // 6)
    customer_ids = [f"CUST{i:05d}" for i in range(n_customers)]
    customers    = np.random.choice(customer_ids, size=n_rows, replace=True)

    base_date = datetime(2023, 1, 1)
    timestamps = [
        base_date + timedelta(
            days=int(np.random.randint(0, 730)),
            hours=int(np.random.randint(7, 23)),
            minutes=int(np.random.randint(0, 60)),
        )
        for _ in range(n_rows)
    ]

    categories     = np.random.choice(CATEGORIES, size=n_rows)
    prices         = np.array([np.random.randint(*PRICE_BANDS[c]) for c in categories], dtype=float)
    cost_ratios    = np.array([COST_RATIO[c] for c in categories])
    cogs           = np.round(prices * cost_ratios, 2)
    delivery_cost  = np.clip(np.round(prices * np.random.uniform(0.03, 0.10, n_rows)), 30, 400)
    delivery_charge= np.clip(np.round(prices * np.random.uniform(0.05, 0.12, n_rows)), 40, 500)

    discount_mask  = np.random.random(n_rows) < 0.30
    discount_pct   = np.where(discount_mask, np.random.uniform(0.05, 0.22, n_rows), 0.0)
    discount_amt   = np.round(prices * discount_pct)
    final_price    = prices - discount_amt
    payment_amount = np.round(final_price + delivery_charge)

    gross_profit   = payment_amount - cogs - delivery_cost
    gross_margin   = np.where(payment_amount > 0, gross_profit / payment_amount * 100, 0)

    review_scores  = np.random.choice([1,2,3,4,5], size=n_rows, p=[0.05,0.08,0.12,0.30,0.45])
    order_statuses = np.random.choice(STATUSES, size=n_rows, p=STATUS_W)

    # Customer-level attributes (consistent per customer)
    cust_gender = {c: np.random.choice(GENDERS, p=[0.48, 0.49, 0.03]) for c in customer_ids}
    cust_age    = {c: np.random.choice(AGE_GROUPS, p=[0.18, 0.32, 0.28, 0.14, 0.08]) for c in customer_ids}
    cust_device = {c: np.random.choice(DEVICES, p=[0.62, 0.30, 0.08]) for c in customer_ids}
    cust_city   = {c: np.random.choice(CITIES) for c in customer_ids}

    df = pd.DataFrame({
        "customer_id":     customers,
        "order_date":      pd.to_datetime(timestamps),
        "category":        categories,
        "price":           prices,
        "cogs":            cogs,
        "delivery_cost":   delivery_cost,
        "delivery_charge": delivery_charge,
        "discount_pct":    np.round(discount_pct * 100, 1),
        "discount_amount": discount_amt,
        "payment_amount":  payment_amount,
        "gross_profit":    gross_profit,
        "gross_margin_pct":np.round(gross_margin, 1),
        "review_score":    review_scores,
        "order_status":    order_statuses,
        "gender":          [cust_gender[c] for c in customers],
        "age_group":       [cust_age[c] for c in customers],
        "device":          [cust_device[c] for c in customers],
        "city":            [cust_city[c] for c in customers],
        "year":            pd.to_datetime(timestamps).year,
        "month":           pd.to_datetime(timestamps).month,
        "month_name":      [t.strftime("%b %Y") for t in timestamps],
        "day_of_week":     [t.strftime("%A") for t in timestamps],
        "hour":            [t.hour for t in timestamps],
    })
    return df


# ── RFM Segmentation ─────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df[df["order_status"] != "cancelled"]
        .groupby("customer_id")
        .agg(
            Recency   = ("order_date",      lambda x: (snapshot - x.max()).days),
            Frequency = ("order_date",      "count"),
            Monetary  = ("payment_amount",  "sum"),
            AvgRating = ("review_score",    "mean"),
            City      = ("city",            "first"),
            Gender    = ("gender",          "first"),
            AgeGroup  = ("age_group",       "first"),
            Device    = ("device",          "first"),
            LastCat   = ("category",        "last"),
        )
        .reset_index()
    )

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km        = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values()
    labels = ["Budget Buyers", "Regular Customers", "Loyal Customers", "Top Customers"]
    label_map = {cid: labels[i] for i, cid in enumerate(cluster_means.index)}
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    # RFM score (1-5 per dimension, combined)
    rfm["R_score"] = pd.qcut(rfm["Recency"],   5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm


# ── Churn Model ───────────────────────────────────────────────────────────────
def train_churn_model(rfm: pd.DataFrame):
    rfm = rfm.copy()
    rfm["AtRisk"] = (rfm["Recency"] > 180).astype(int)
    feats = ["Recency", "Frequency", "Monetary", "AvgRating"]
    X, y  = rfm[feats].fillna(0), rfm["AtRisk"]
    if len(y.unique()) < 2 or len(X) < 20:
        return None, feats
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, feats


# ── Behavior Features ─────────────────────────────────────────────────────────
def compute_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer behavioral metrics."""
    b = (
        df.groupby("customer_id")
        .agg(
            TotalOrders    = ("order_date",      "count"),
            TotalSpend     = ("payment_amount",  "sum"),
            AvgOrderValue  = ("payment_amount",  "mean"),
            AvgDiscount    = ("discount_pct",    "mean"),
            FavCategory    = ("category",        lambda x: x.mode()[0]),
            AvgRating      = ("review_score",    "mean"),
            CancelRate     = ("order_status",    lambda x: (x == "cancelled").mean() * 100),
            PrefDevice     = ("device",          lambda x: x.mode()[0]),
            PrefHour       = ("hour",            "median"),
        )
        .reset_index()
    )
    b["AvgOrderValue"] = b["AvgOrderValue"].round(0)
    b["AvgDiscount"]   = b["AvgDiscount"].round(1)
    b["AvgRating"]     = b["AvgRating"].round(2)
    b["CancelRate"]    = b["CancelRate"].round(1)
    return b


# ── P&L Summary ───────────────────────────────────────────────────────────────
def compute_pnl(df: pd.DataFrame) -> dict:
    d = df[df["order_status"] != "cancelled"]
    revenue       = d["payment_amount"].sum()
    cogs          = d["cogs"].sum()
    delivery_cost = d["delivery_cost"].sum()
    discount_cost = d["discount_amount"].sum()
    gross_profit  = d["gross_profit"].sum()
    # Estimated operating expenses
    marketing     = revenue * 0.08
    ops_overhead  = revenue * 0.05
    net_profit    = gross_profit - marketing - ops_overhead

    monthly = (
        d.groupby("month_name")
        .agg(Revenue=("payment_amount","sum"), COGS=("cogs","sum"),
             GrossProfit=("gross_profit","sum"), Orders=("order_date","count"))
        .reset_index()
    )
    monthly["NetProfit"] = monthly["GrossProfit"] - monthly["Revenue"] * 0.13

    cat_pnl = (
        d.groupby("category")
        .agg(Revenue=("payment_amount","sum"), COGS=("cogs","sum"),
             GrossProfit=("gross_profit","sum"), Orders=("order_date","count"))
        .reset_index()
    )
    cat_pnl["Margin%"] = (cat_pnl["GrossProfit"] / cat_pnl["Revenue"] * 100).round(1)
    cat_pnl = cat_pnl.sort_values("GrossProfit", ascending=False)

    return {
        "revenue":       revenue,
        "cogs":          cogs,
        "delivery_cost": delivery_cost,
        "discount_cost": discount_cost,
        "gross_profit":  gross_profit,
        "marketing":     marketing,
        "ops_overhead":  ops_overhead,
        "net_profit":    net_profit,
        "gross_margin":  gross_profit / revenue * 100 if revenue else 0,
        "net_margin":    net_profit / revenue * 100 if revenue else 0,
        "monthly":       monthly,
        "cat_pnl":       cat_pnl,
    }
