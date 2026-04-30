import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_olist_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset mimicking the Olist Brazilian
    e-commerce dataset structure.  Columns match what ml_engine.py expects:
        customer_unique_id, order_purchase_timestamp, price, freight_value,
        product_category_name, review_score, order_status, payment_value
    """
    np.random.seed(seed)

    categories = [
        "cama_mesa_banho", "beleza_saude", "esporte_lazer", "moveis_decoracao",
        "utilidades_domesticas", "informatica_acessorios", "cool_stuff",
        "ferramentas_jardim", "automotivo", "brinquedos",
        "eletronicos", "perfumaria", "papelaria", "pet_shop", "alimentos",
    ]
    statuses = ["delivered", "shipped", "processing", "canceled"]
    status_weights = [0.80, 0.10, 0.07, 0.03]

    n_customers = max(50, n_rows // 8)
    customer_ids = [f"cust_{i:05d}" for i in range(n_customers)]

    # Each customer gets 1-15 orders
    customers = np.random.choice(customer_ids, size=n_rows, replace=True)

    base_date = datetime(2018, 1, 1)
    days_range = 700  # ~2 years of data
    timestamps = [
        base_date + timedelta(days=int(np.random.randint(0, days_range)),
                              hours=int(np.random.randint(0, 24)),
                              minutes=int(np.random.randint(0, 60)))
        for _ in range(n_rows)
    ]

    prices = np.round(np.random.lognormal(mean=4.2, sigma=0.9, size=n_rows), 2)
    prices = np.clip(prices, 5.0, 2500.0)

    freight = np.round(prices * np.random.uniform(0.05, 0.25, size=n_rows), 2)
    payment_values = np.round(prices + freight, 2)

    review_scores = np.random.choice([1, 2, 3, 4, 5], size=n_rows,
                                     p=[0.07, 0.08, 0.10, 0.25, 0.50])

    df = pd.DataFrame({
        "customer_unique_id": customers,
        "order_purchase_timestamp": timestamps,
        "price": prices,
        "freight_value": freight,
        "product_category_name": np.random.choice(categories, size=n_rows),
        "review_score": review_scores,
        "order_status": np.random.choice(statuses, size=n_rows, p=status_weights),
        "payment_value": payment_values,
    })

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df
