"""
shap_page.py
=============
Drop this into your Dashboard.py as a new tab/page, or run standalone.

To add to your existing Dashboard.py, paste the contents of the
"── SHAP Page ──" section and add it as a new st.tab().

Standalone run:
    streamlit run shap_page.py

Requires:
    pip install shap plotly streamlit pandas joblib xgboost
"""

import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Import your existing pipeline ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shap_explainer import (
    build_shap_explainer,
    explain_customer,
    global_feature_importance,
    generate_plain_english_summary,
    FEATURE_LABELS,
)

# Reuse helpers from your main Dashboard
MODEL_FEATURES = [
    "Recency", "Frequency", "Monetary", "Avg_Order_Value",
    "Customer_Age", "Time_Since_Last_Purchase",
    "Unique_Products", "Total_Quantity", "Purchase_Velocity",
]


# ── Page config (only needed if running standalone) ───────────────────────────
st.set_page_config(
    page_title="Why did they churn? — SHAP Explainer",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.factor-bar-positive { background: linear-gradient(90deg, #ef444433, #ef4444); height:28px; border-radius:4px; }
.factor-bar-negative  { background: linear-gradient(90deg, #10b98133, #10b981); height:28px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Churn Explainer")
    st.markdown("---")
    uploaded = st.file_uploader("Upload order CSV", type=["csv"])
    st.markdown("---")
    st.caption("Uses SHAP to explain why each customer is predicted to churn or stay.")


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_churn_model():
    path = Path(__file__).resolve().parent / "models" / "churn_model.pkl"
    if not path.exists():
        return None
    model = joblib.load(path)
    if isinstance(model, tuple):
        model = model[0]
    return model


# ── Data helpers (same as Dashboard.py) ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found.")
    df.rename(columns={date_col: "InvoiceDate"}, inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    for variants, target in [
        (["quantity", "qty", "units"],                     "Quantity"),
        (["unitprice", "unit_price", "price", "amount"],   "UnitPrice"),
        (["invoiceno", "invoice_no", "order_id"],          "InvoiceNo"),
        (["stockcode", "stock_code", "sku", "product_id"], "StockCode"),
    ]:
        for c in df.columns:
            if c.lower().replace(" ", "").replace("_", "") in [v.replace("_", "") for v in variants]:
                df.rename(columns={c: target}, inplace=True)
                break
    if "Quantity"  not in df.columns: df["Quantity"]  = 1
    if "UnitPrice" not in df.columns: df["UnitPrice"] = 0
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    if "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = df["InvoiceDate"].dt.date.astype(str) + "_" + df["CustomerID"]
    if "StockCode" not in df.columns:
        df["StockCode"] = "UNKNOWN"
    return df


def auto_detect_dates(df):
    min_d  = df["InvoiceDate"].min()
    max_d  = df["InvoiceDate"].max()
    total  = (max_d - min_d).days
    cutoff = min_d + pd.Timedelta(days=int(total * 0.65))
    return cutoff, max_d


def build_features(past_data, cutoff_date):
    rfm = past_data.groupby("CustomerID").agg(
        Recency             =("InvoiceDate",  lambda x: (cutoff_date - x.max()).days),
        Frequency           =("InvoiceNo",    "nunique"),
        Monetary            =("TotalPrice",   "sum"),
        Avg_Order_Value     =("TotalPrice",   "mean"),
        Unique_Products     =("StockCode",    "nunique"),
        Total_Quantity      =("Quantity",     "sum"),
        First_Purchase_Date =("InvoiceDate",  "min"),
        Num_Invoices        =("InvoiceNo",    "nunique"),
    ).reset_index()
    rfm["Customer_Age"]             = (cutoff_date - rfm["First_Purchase_Date"]).dt.days
    rfm["Time_Since_Last_Purchase"] = rfm["Recency"]
    rfm["Purchase_Velocity"]        = (
        rfm["Num_Invoices"] / (rfm["Customer_Age"] / 30)
    ).replace([float("inf")], 0).fillna(0)
    return rfm[["CustomerID"] + MODEL_FEATURES]


# ── SHAP chart helpers ────────────────────────────────────────────────────────

def waterfall_chart(explanation, customer_id, churn_prob):
    """Plotly waterfall chart showing SHAP contributions."""
    factors = explanation["factors"][:8]  # top 8 for readability

    labels  = [f["label"] for f in factors]
    values  = [f["shap_value"] for f in factors]
    colors  = ["#ef4444" if v > 0 else "#10b981" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}<br>SHAP impact: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Customer {customer_id} — Churn probability: {churn_prob*100:.1f}%",
            font_size=14,
        ),
        xaxis_title="Impact on churn risk  ← reduces risk  |  increases risk →",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=2,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis_gridcolor="rgba(255,255,255,0.08)",
    )
    return fig


def global_importance_chart(importance_df):
    """Bar chart of global feature importance."""
    fig = go.Figure(go.Bar(
        x=importance_df["mean_abs_shap"],
        y=importance_df["label"],
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.4f}" for v in importance_df["mean_abs_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="What drives churn across ALL customers",
        xaxis_title="Mean |SHAP| value (average impact on churn prediction)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis_gridcolor="rgba(255,255,255,0.08)",
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔍 Why is this customer going to churn?")
st.markdown(
    "SHAP (SHapley Additive exPlanations) breaks down exactly what is pushing "
    "each customer towards or away from churning — in plain English."
)

churn_model = load_churn_model()
if churn_model is None:
    st.error(
        "Churn model not found at `models/churn_model.pkl`. "
        "Run `python train_churn_model.py` first."
    )
    st.stop()

if uploaded is None:
    st.info("Upload your order CSV from the sidebar to get started.")
    st.stop()

# ── Data pipeline ─────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_and_clean(uploaded.read())

cutoff_date, _ = auto_detect_dates(df)
past_data   = df[df["InvoiceDate"] <= cutoff_date]

with st.spinner("Building customer features..."):
    features_df = build_features(past_data, cutoff_date)

X = features_df[MODEL_FEATURES].fillna(0)

with st.spinner("Computing churn probabilities..."):
    churn_probs = churn_model.predict_proba(X)[:, 1]
    features_df["Churn_Risk"] = churn_probs

with st.spinner("Building SHAP explainer (first run takes ~30 seconds)..."):
    background = X.sample(min(300, len(X)), random_state=42)
    explainer  = build_shap_explainer(churn_model, background)

st.success(f"Ready — {len(features_df):,} customers loaded.")
st.markdown("---")


# ── Section 1: Global importance ──────────────────────────────────────────────
st.subheader("What drives churn across your whole customer base?")
st.caption(
    "The longer the bar, the more that factor matters for predicting churn "
    "across ALL your customers — not just one."
)

with st.spinner("Computing global SHAP importance..."):
    importance_df = global_feature_importance(explainer, X, top_n=9)

st.plotly_chart(global_importance_chart(importance_df), use_container_width=True)

# Plain-English summary of top driver
top = importance_df.iloc[0]
st.info(
    f"**The single biggest driver of churn in your customer base is: "
    f"{top['label']}**\n\n"
    f"This factor has an average SHAP impact of {top['mean_abs_shap']:.4f} — "
    f"meaning it moves the churn probability more than any other variable. "
    f"Focus your retention strategy here first."
)

st.markdown("---")


# ── Section 2: Individual customer explainer ─────────────────────────────────
st.subheader("Why is a specific customer at risk?")
st.caption("Select any customer to see a breakdown of what's driving their churn risk.")

# Sort by churn risk descending for easy access to at-risk customers
display_df = features_df[["CustomerID", "Churn_Risk", "Recency", "Frequency", "Monetary"]].copy()
display_df["Churn_Risk_Pct"] = (display_df["Churn_Risk"] * 100).round(1).astype(str) + "%"
display_df = display_df.sort_values("Churn_Risk", ascending=False)

col_pick, col_info = st.columns([1, 2])

with col_pick:
    customer_ids = display_df["CustomerID"].tolist()
    selected_id  = st.selectbox(
        "Choose customer",
        customer_ids,
        format_func=lambda x: f"{x}  ({display_df[display_df['CustomerID']==x]['Churn_Risk_Pct'].values[0]} churn risk)"
    )

    if selected_id:
        row     = features_df[features_df["CustomerID"] == selected_id].iloc[0]
        prob    = float(row["Churn_Risk"])

        if prob > 0.65:
            st.error(f"**{prob*100:.1f}% churn risk** — high priority")
        elif prob > 0.4:
            st.warning(f"**{prob*100:.1f}% churn risk** — monitor closely")
        else:
            st.success(f"**{prob*100:.1f}% churn risk** — looks healthy")

        st.metric("Days since last order", int(row["Recency"]))
        st.metric("Total orders",          int(row["Frequency"]))
        st.metric("Total spent",           f"₹{row['Monetary']:,.0f}")

with col_info:
    if selected_id:
        customer_features = row[MODEL_FEATURES]
        explanation       = explain_customer(explainer, customer_features, model_type="churn")
        summary           = generate_plain_english_summary(explanation, selected_id, prob)

        st.markdown(f"**Plain-English summary:**\n\n{summary}")
        st.markdown("")
        st.plotly_chart(
            waterfall_chart(explanation, selected_id, prob),
            use_container_width=True
        )

st.markdown("---")


# ── Section 3: Top at-risk customers with explanations ───────────────────────
st.subheader("Top 10 customers most at risk — and why")
st.caption(
    "Red factors are pushing them toward churning. "
    "Green factors are keeping them. "
    "Focus on the longest red bars."
)

top10 = display_df.head(10)

for _, row_data in top10.iterrows():
    cid   = row_data["CustomerID"]
    prob  = float(row_data["Churn_Risk"])
    color = "🔴" if prob > 0.65 else "🟡"

    with st.expander(f"{color} Customer {cid} — {prob*100:.1f}% churn risk"):
        cust_row    = features_df[features_df["CustomerID"] == cid].iloc[0]
        explanation = explain_customer(explainer, cust_row[MODEL_FEATURES], model_type="churn")
        summary     = generate_plain_english_summary(explanation, cid, prob)

        st.markdown(summary)

        # Simple bar table — no plotly overhead for 10 rows
        top_factors = explanation["factors"][:5]
        for f in top_factors:
            label   = f["label"]
            impact  = f["shap_value"]
            bar_pct = min(int(abs(impact) / 0.5 * 100), 100)
            bar_col = "#ef4444" if impact > 0 else "#10b981"
            sign    = "▲ Increases churn risk" if impact > 0 else "▼ Reduces churn risk"

            st.markdown(
                f"<div style='margin:4px 0'>"
                f"<div style='font-size:12px;opacity:0.7;margin-bottom:2px'>{label} &nbsp;·&nbsp; {sign}</div>"
                f"<div style='background:{bar_col};opacity:0.85;width:{bar_pct}%;height:10px;border-radius:3px'></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Action recommendation
        st.markdown("")
        top_risk = [f for f in explanation["factors"] if f["direction"] == "increases_risk"]
        if top_risk:
            primary_issue = top_risk[0]["label"].lower()
            if "days since" in primary_issue or "recency" in primary_issue.lower():
                action = "Send a personalised win-back email with a small discount — they've simply gone quiet."
            elif "order" in primary_issue or "frequency" in primary_issue.lower():
                action = "Offer a bundle deal or loyalty reward to encourage their next purchase."
            elif "variety" in primary_issue or "product" in primary_issue.lower():
                action = "Recommend new product categories they haven't tried yet."
            else:
                action = "Reach out personally — their engagement signals suggest they're drifting away."
            st.success(f"**Suggested action:** {action}")

st.markdown("---")


# ── Section 4: Export SHAP explanations ──────────────────────────────────────
st.subheader("Export explanations for your team")

with st.spinner("Building explanation export..."):
    export_rows = []
    sample = features_df.head(min(200, len(features_df)))  # cap at 200 for speed

    for _, row_data in sample.iterrows():
        cid  = row_data["CustomerID"]
        prob = float(row_data["Churn_Risk"])
        exp  = explain_customer(explainer, row_data[MODEL_FEATURES], model_type="churn")
        top3 = [f["label"] for f in exp["factors"][:3]]
        summary = generate_plain_english_summary(exp, cid, prob)

        export_rows.append({
            "CustomerID":           cid,
            "Churn_Risk_%":         round(prob * 100, 1),
            "Top_Risk_Factor_1":    top3[0] if len(top3) > 0 else "",
            "Top_Risk_Factor_2":    top3[1] if len(top3) > 1 else "",
            "Top_Risk_Factor_3":    top3[2] if len(top3) > 2 else "",
            "Plain_English_Reason": summary,
        })

    export_df = pd.DataFrame(export_rows).sort_values("Churn_Risk_%", ascending=False)

st.download_button(
    "⬇️ Download SHAP explanations (CSV)",
    data=export_df.to_csv(index=False).encode(),
    file_name="churn_explanations.csv",
    mime="text/csv",
    use_container_width=False,
)
st.caption(
    "This CSV contains the top 3 churn risk factors and a plain-English reason "
    "for every customer — ready to share with your marketing team or upload to your CRM."
)