import pandas as pd
import numpy as np
from datetime import datetime, timedelta


CATEGORIES = [
    "clothing_fashion", "electronics", "home_kitchen", "beauty_personal_care",
    "sports_fitness", "books_stationery", "toys_games", "groceries_food",
    "mobile_accessories", "furniture_decor", "health_wellness", "automotive",
    "jewellery_watches", "baby_products", "pet_supplies",
]

PRICE_BANDS = {
    "clothing_fashion":       (299,   4999),
    "electronics":            (999,  49999),
    "home_kitchen":           (199,   9999),
    "beauty_personal_care":   (99,    2999),
    "sports_fitness":         (499,  14999),
    "books_stationery":       (99,    1499),
    "toys_games":             (199,   4999),
    "groceries_food":         (49,    1999),
    "mobile_accessories":     (199,   4999),
    "furniture_decor":        (999,  29999),
    "health_wellness":        (149,   3999),
    "automotive":             (299,   9999),
    "jewellery_watches":      (499,  24999),
    "baby_products":          (299,   5999),
    "pet_supplies":           (199,   3999),
}

STATUSES       = ["delivered", "shipped", "processing", "cancelled"]
STATUS_WEIGHTS = [0.80, 0.10, 0.07, 0.03]

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]


def generate_mock_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic Indian e-commerce data. All prices in INR."""
    np.random.seed(seed)

    n_customers  = max(60, n_rows // 7)
    customer_ids = [f"CUST{i:05d}" for i in range(n_customers)]
    customers    = np.random.choice(customer_ids, size=n_rows, replace=True)

    base_date  = datetime(2023, 1, 1)
    timestamps = [
        base_date + timedelta(
            days=int(np.random.randint(0, 700)),
            hours=int(np.random.randint(8, 23)),
            minutes=int(np.random.randint(0, 60)),
        )
        for _ in range(n_rows)
    ]

    categories = np.random.choice(CATEGORIES, size=n_rows)
    prices     = np.array([np.random.randint(*PRICE_BANDS[c]) for c in categories], dtype=float)
    delivery   = np.clip(np.round(prices * np.random.uniform(0.05, 0.15, n_rows)), 40, 500)

    discount_mask    = np.random.random(n_rows) < 0.30
    discount_pct     = np.where(discount_mask, np.random.uniform(0.05, 0.20, n_rows), 0.0)
    discounted_price = np.round(prices * (1 - discount_pct))
    payment_amounts  = np.round(discounted_price + delivery)

    review_scores = np.random.choice([1, 2, 3, 4, 5], size=n_rows,
                                     p=[0.05, 0.08, 0.12, 0.30, 0.45])

    df = pd.DataFrame({
        "customer_id":     customers,
        "order_date":      timestamps,
        "category":        categories,
        "price":           prices,
        "delivery_charge": delivery,
        "discount_amount": np.round(prices * discount_pct),
        "payment_amount":  payment_amounts,
        "review_score":    review_scores,
        "order_status":    np.random.choice(STATUSES, size=n_rows, p=STATUS_WEIGHTS),
        "city":            np.random.choice(CITIES, size=n_rows),
    })

    df["order_date"] = pd.to_datetime(df["order_date"])
    return df
