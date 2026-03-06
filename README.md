# 📈 Revenue Prediction & Customer Retention System
### Built for D2C Startups | Powered by XGBoost + RFM Analytics

---

## The Problem

D2C brands invest heavily in acquisition but rarely know:
- **Which customers will generate revenue** in the next 6 months
- **Which customers are about to churn** — before it's too late to act
- **Where to focus retention spend** for maximum ROI

Most analytics tools give you dashboards. This system gives you **decisions**.

---

## What You Get

| Deliverable | Description |
|---|---|
|  **6-Month Revenue Forecast** | Predicted future revenue per customer |
|  **Churn Risk Score** | Probability each customer churns (0–1) |
|  **LTV Segments** | High / Mid / Low value customer tiers |
|  **Feature Importance Report** | What's actually driving your revenue |
|  **Exportable CSV** | Actionable list ready for your CRM or email tool |

---

## Sample Output

```
CustomerID  Predicted_6M_Revenue  Churn_Risk  Segment
12345       £1,842                0.12        High Value
67890       £290                  0.78        At Risk
11223       £0                    0.95        Lost
```

> A D2C brand using this system identified their top 20% of customers driving 68% of revenue — and cut churn in that segment by retargeting them 2 weeks earlier.

---

## How It Works

```
Raw Transaction Data (CSV)
        ↓
  Data Cleaning & Validation
  (removes returns, nulls, bad prices)
        ↓
  Feature Engineering
  ┌─────────────────────────────┐
  │  RFM (Recency, Frequency,   │
  │  Monetary)                  │
  │  + Customer Age             │
  │  + Purchase Velocity        │
  │  + Product Breadth          │
  └─────────────────────────────┘
        ↓
  Time-Based Train/Test Split
  (no data leakage)
        ↓
  XGBoost + RandomizedSearchCV
  (75 combinations, 5-fold CV)
        ↓
  Revenue Predictions + Churn Labels
        ↓
  Exportable Report
```

---

## Tech Stack

- **Python 3.10+**
- **XGBoost** — gradient boosted trees for revenue prediction
- **scikit-learn** — RandomizedSearchCV, metrics, preprocessing
- **pandas** — data pipeline
- **joblib** — model serialisation

---

## Project Structure

```
├── Dataset/                    # Raw transaction data (gitignored)
│   └── Raw_Data.csv
├── models/                     # Trained model files + metrics JSON
│   ├── churn_model.pkl
│   └── revenue_model.pkl
├── Notebooks/                  # Exploratory analysis
│   └── Data_Cleaning.ipynb
├── scripts/                    # Training entry points
│   ├── train_churn_model.py
│   └── train_revenue_model.py
├── src/                        # Core pipeline modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── ltv_model.py
├── venv/                       # Virtual environment (gitignored)
├── client_report_sample.html   # Sample client-facing output report
├── main.py                     # Single entry point — run everything
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/revenue-retention-system
cd revenue-retention-system
pip install -r requirements.txt

# 2. Add your data
cp your_transactions.csv Dataset/Raw_Data.csv

# 3. Run the full pipeline
python main.py
```

**Expected CSV format:**

| Column | Type | Example |
|---|---|---|
| CustomerID | int | 12345 |
| InvoiceNo | string | 536365 |
| InvoiceDate | datetime | 2011-01-06 08:26:00 |
| Quantity | int | 6 |
| UnitPrice | float | 2.55 |
| StockCode | string | 85123A |

---

## Model Performance (on UCI Online Retail Dataset)

| Metric | Revenue Model |
|---|---|
| MAE | ~£185 |
| RMSE | ~£320 |
| R² | ~0.81 |
| CV Best RMSE | ~£298 |

> Performance varies by dataset. Tuned via RandomizedSearchCV (75 iterations, 5-fold CV).

---

## For D2C Founders

You don't need to understand the model. You need to know:

1. **Upload your Shopify/WooCommerce order export** (standard CSV format)
2. **Receive a prioritised customer list** — who to retain, who to upsell, who to let go
3. **Feed it into Klaviyo, Mailchimp, or your CRM** in one click

Typical engagement: **one-time analysis** or **monthly refresh** as new orders come in.

---

## About

Built by [Your Name] — data scientist specialising in D2C customer analytics.

📧 bhaveshjangra889@gmail.com | 🌐https://github.com/Bhavesh2123  | 💼 https://www.linkedin.com/in/bhavesh-jangra-a39a90292/