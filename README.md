# 🛍️ ShopIndia Intelligence Platform

> **Customer Segmentation & Behaviour Analysis** — A dual-portal Streamlit application for Indian e-commerce analytics.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-green)](https://plotly.com)

---

## 📌 What This App Does

ShopIndia is a **self-contained analytics platform** with two completely separate portals:

| Portal | Who Uses It | What They See |
|--------|-------------|---------------|
| 👤 **Customer Portal** | Shoppers | Personalised shop, order history, spending behaviour, profile |
| 📊 **Manager Portal** | Store managers | Segmentation, behaviour analysis, P&L, churn risk, advanced analytics |

---

## ✨ Key Features

### 👤 Customer Portal
- 🏠 **Home Dashboard** — personal KPIs, spending trend, category breakdown
- 🛒 **Shop** — browse products filtered by segment, search, sort
- 📦 **My Orders** — full order history with filters
- 👤 **My Profile** — behaviour summary, spending patterns by day of week

### 📊 Manager Portal
- 📊 **Dashboard** — revenue trends, order heatmap, city-wise performance
- 👥 **Customer Segmentation** — RFM KMeans clustering (4 segments), scatter plots, segment profiles
- 🧠 **Behaviour Analysis** — time patterns, device/gender/age breakdown, category behaviour
- 💰 **Profit & Loss** — waterfall P&L, monthly trends, category margins, cost breakdown
- 📈 **Advanced Analytics** — repeat vs new customers, cohort analysis, geo treemap
- ⚠️ **Churn Risk** — Random Forest model, risk gauge, interactive predictor, high-risk list
- 🛒 **Product Analytics** — category deep dive, cancellation rates, performance matrix

---

## 🚀 Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/ShopIndia-Intelligence.git
cd ShopIndia-Intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🔑 Demo Login Credentials

| Role | Email | Password |
|------|-------|----------|
| 👤 Customer | customer@demo.com | customer123 |
| 📊 Manager  | manager@demo.com  | manager123  |

---

## 📁 Project Structure

```
ShopIndia-Intelligence/
├── app.py                    # Main entry point, login, routing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── pages/
│   ├── __init__.py
│   ├── customer_portal.py    # Full customer experience
│   └── manager_portal.py     # Full manager analytics
└── utils/
    ├── __init__.py
    ├── auth.py               # SQLite login system
    └── data.py               # Data generation + all ML models
```

---

## 🧠 Machine Learning Models

### 1. RFM Segmentation (KMeans Clustering)
- Computes **Recency**, **Frequency**, **Monetary** per customer
- KMeans clusters into 4 segments:
  - 🥇 **Top Customers** — high spend, frequent, recent
  - 🥈 **Loyal Customers** — frequent buyers
  - 🥉 **Regular Customers** — average behaviour
  - 💡 **Budget Buyers** — low spend, infrequent

### 2. Churn Prediction (Random Forest)
- Labels customers at risk if last order > 180 days ago
- Features: Recency, Frequency, Monetary, Avg Rating
- Outputs probability score + risk level (High / Medium / Low)
- Shows feature importance chart

---

## 📊 Data

- **Synthetic data** is auto-generated on first run (4,000 orders, ~570 customers)
- All prices in **Indian Rupees (₹)**
- Covers 15 product categories, 10 Indian cities
- Includes gender, age group, device, city demographics
- Full P&L computed with realistic cost ratios per category

**No CSV files needed — the app runs immediately.**

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select `app.py` as the main file
5. Click **Deploy** — done in ~60 seconds

> **Free tier note:** All heavy computation (`generate_data`, `compute_rfm`, `train_churn_model`) is wrapped in `@st.cache_data` so it runs **only once** per session. Subsequent page changes are instant.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | KMeans clustering, Random Forest |
| `plotly` | Interactive charts |

---

## 📸 Screenshots

### Landing / Login
- Clean gradient login screen with demo credentials shown

### Customer Home
- Personal KPI cards + spending trend + category pie chart

### Manager Segmentation
- RFM scatter plot + segment distribution + demographic drill-down

### Profit & Loss
- Waterfall P&L chart + monthly revenue vs profit + category margin table

### Churn Risk
- Risk gauge + feature importance + high-risk customer list

---

## 📄 License

MIT License — free to use and modify.

---

## 👨‍💻 Author

Built with ❤️ for Indian e-commerce analytics.

*Powered by Streamlit + scikit-learn + Plotly*
