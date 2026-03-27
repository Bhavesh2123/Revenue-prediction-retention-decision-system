"""
D2C Revenue & Churn Dashboard
==============================
Run with:
    streamlit run Dashboard.py

Requirements:
    pip install streamlit pandas xgboost scikit-learn plotly joblib
"""

import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="D2C Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal safe CSS (no background overrides) ────────────────────────────────
st.markdown("""
<style>
.kpi-label { font-size:13px; font-weight:600; opacity:0.6; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }
.kpi-value { font-size:36px; font-weight:700; line-height:1.1; margin-bottom:2px; }
.kpi-sub   { font-size:12px; opacity:0.5; }
.tip-green  { border-left:4px solid #10b981; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; background:rgba(16,185,129,0.08); }
.tip-amber  { border-left:4px solid #f59e0b; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; background:rgba(245,158,11,0.08); }
.tip-red    { border-left:4px solid #ef4444; padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0; background:rgba(239,68,68,0.08); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(val, sym="₹"):
    if val >= 1_000_000:
        return f"{sym}{val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{sym}{val/1_000:.1f}K"
    return f"{sym}{val:.0f}"

def fmt_pct(val):
    return f"{val*100:.1f}%"


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()

    # Flexible date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found. Need a column with 'date' in the name.")
    df.rename(columns={date_col: 'InvoiceDate'}, inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)

    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip()

    # Flexible column matching
    for variants, target in [
        (['quantity', 'qty', 'units'],                    'Quantity'),
        (['unitprice', 'unit_price', 'price', 'amount'],  'UnitPrice'),
        (['invoiceno', 'invoice_no', 'order_id','orderid'],'InvoiceNo'),
        (['stockcode', 'stock_code', 'sku', 'product_id'],'StockCode'),
    ]:
        for c in df.columns:
            if c.lower().replace(' ', '').replace('_', '') in [v.replace('_','') for v in variants]:
                df.rename(columns={c: target}, inplace=True)
                break

    if 'UnitPrice' not in df.columns and 'TotalPrice' in df.columns:
        df['UnitPrice'] = df['TotalPrice'] / df.get('Quantity', 1)
    if 'Quantity'  not in df.columns: df['Quantity']  = 1
    if 'UnitPrice' not in df.columns: df['UnitPrice'] = 0

    df = df[df['Quantity']  > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    if 'InvoiceNo'  not in df.columns:
        df['InvoiceNo']  = df['InvoiceDate'].dt.date.astype(str) + '_' + df['CustomerID']
    if 'StockCode'  not in df.columns:
        df['StockCode']  = 'UNKNOWN'

    return df


def auto_detect_dates(df, obs_fraction=0.65):
    min_d  = df['InvoiceDate'].min()
    max_d  = df['InvoiceDate'].max()
    total  = (max_d - min_d).days
    cutoff = min_d + pd.Timedelta(days=int(total * obs_fraction))
    return cutoff, max_d


def build_features(past_data, cutoff_date):
    rfm = past_data.groupby('CustomerID').agg(
        Recency              =('InvoiceDate',  lambda x: (cutoff_date - x.max()).days),
        Frequency            =('InvoiceNo',    'nunique'),
        Monetary             =('TotalPrice',   'sum'),
        Avg_Order_Value      =('TotalPrice',   'mean'),
        Unique_Products      =('StockCode',    'nunique'),
        Total_Quantity       =('Quantity',     'sum'),
        First_Purchase_Date  =('InvoiceDate',  'min'),
        Num_Invoices         =('InvoiceNo',    'nunique'),
    ).reset_index()

    rfm['Customer_Age']              = (cutoff_date - rfm['First_Purchase_Date']).dt.days
    rfm['Time_Since_Last_Purchase']  = rfm['Recency']
    rfm['Purchase_Velocity']         = (
        rfm['Num_Invoices'] / (rfm['Customer_Age'] / 30)
    ).replace([float('inf')], 0).fillna(0)

    return rfm[[
        'CustomerID','Recency','Frequency','Monetary',
        'Avg_Order_Value','Customer_Age','Time_Since_Last_Purchase',
        'Unique_Products','Total_Quantity','Purchase_Velocity'
    ]]


@st.cache_resource(show_spinner=False)
def load_models():
    base        = Path(__file__).resolve().parent / "models"
    rev_path    = base / "revenue_model.pkl"
    churn_path  = base / "churn_model.pkl"
    rev_model   = joblib.load(rev_path)   if rev_path.exists()   else None
    churn_model = joblib.load(churn_path) if churn_path.exists() else None
    return rev_model, churn_model


MODEL_FEATURES = [
    'Recency','Frequency','Monetary','Avg_Order_Value',
    'Customer_Age','Time_Since_Last_Purchase',
    'Unique_Products','Total_Quantity','Purchase_Velocity'
]


def run_predictions(features_df, rev_model, churn_model):
    X = features_df[MODEL_FEATURES].fillna(0)
    result = features_df[['CustomerID']].copy()

    if rev_model is not None:
        result['Predicted_Revenue'] = np.clip(rev_model.predict(X), 0, None)
    else:
        result['Predicted_Revenue'] = (
            features_df['Monetary'] * 0.6 +
            features_df['Frequency'] * features_df['Avg_Order_Value'] * 0.1
        ).clip(lower=0).values

    if churn_model is not None:
        result['Churn_Risk'] = churn_model.predict_proba(X)[:, 1]
    else:
        max_rec = features_df['Recency'].max()
        result['Churn_Risk'] = (
            features_df['Recency'].values / max_rec * 0.7 +
            (1 / (features_df['Frequency'].values + 1)) * 0.3
        ).clip(0, 1)

    q75 = result['Predicted_Revenue'].quantile(0.75)
    q40 = result['Predicted_Revenue'].quantile(0.40)

    def segment(row):
        if row['Churn_Risk'] > 0.65:
            return 'At Risk'
        elif row['Predicted_Revenue'] >= q75:
            return 'High Value'
        elif row['Predicted_Revenue'] >= q40:
            return 'Mid Value'
        else:
            return 'Low Value'

    result['Segment'] = result.apply(segment, axis=1)
    result = result.merge(
        features_df[['CustomerID','Recency','Frequency','Monetary','Avg_Order_Value']],
        on='CustomerID'
    )
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 D2C Intelligence")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload your order CSV",
        type=["csv"],
        help="Needs: CustomerID, InvoiceDate, Quantity, UnitPrice (or TotalPrice)"
    )

    st.markdown("---")
    st.markdown("**Expected columns**")
    st.markdown("- `CustomerID`\n- `InvoiceDate`\n- `Quantity`\n- `UnitPrice` or `TotalPrice`\n- `InvoiceNo` *(optional)*\n- `StockCode` / SKU *(optional)*")

    st.markdown("---")
    currency_symbol = st.selectbox("Currency", ["₹", "£", "$", "€", "AED"])

    st.markdown("---")
    models_dir  = Path(__file__).resolve().parent / "models"
    rev_exists  = (models_dir / "revenue_model.pkl").exists()
    churn_exists= (models_dir / "churn_model.pkl").exists()
    st.markdown(f"{'✅' if rev_exists   else '⚠️'} Revenue model")
    st.markdown(f"{'✅' if churn_exists else '⚠️'} Churn model")
    if not rev_exists or not churn_exists:
        st.warning("No trained models found — running in **demo mode** with estimated predictions.")

    st.markdown("---")
    st.caption("Powered by XGBoost + RFM Analytics")


# ── Landing page ──────────────────────────────────────────────────────────────
if uploaded is None:
    st.title("D2C Revenue & Churn Intelligence")
    st.markdown("Upload your order history CSV from the sidebar to get started.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1 — Upload CSV**\n\nYour Shopify / WooCommerce / custom order export")
    with c2:
        st.warning("**Step 2 — Auto Analysis**\n\nRevenue forecast + churn risk scores per customer")
    with c3:
        st.success("**Step 3 — Take Action**\n\nDownload your priority customer list for your CRM")

    st.stop()


# ── Data pipeline ─────────────────────────────────────────────────────────────
with st.spinner("Reading and cleaning your data..."):
    try:
        df = load_and_clean(uploaded.read())
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

cutoff_date, data_end = auto_detect_dates(df)
past_data   = df[df['InvoiceDate'] <= cutoff_date]
future_data = df[df['InvoiceDate'] >  cutoff_date]

with st.spinner("Building customer features..."):
    features_df = build_features(past_data, cutoff_date)

with st.spinner("Running predictions..."):
    rev_model, churn_model = load_models()
    results = run_predictions(features_df, rev_model, churn_model)

# Derived stats
n_customers      = len(results)
total_rev        = results['Predicted_Revenue'].sum()
at_risk          = results[results['Segment'] == 'At Risk']
high_value       = results[results['Segment'] == 'High Value']
mid_value        = results[results['Segment'] == 'Mid Value']
avg_churn        = results['Churn_Risk'].mean()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Revenue & Churn Intelligence")
st.caption(
    f"Data: {past_data['InvoiceDate'].min().date()} → {cutoff_date.date()} · "
    f"{n_customers:,} customers analysed · "
    f"{'Demo mode' if rev_model is None else 'Live model'}"
)
st.markdown("---")


# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="💰 Forecast Revenue (6M)",
        value=fmt_currency(total_rev, currency_symbol),
        delta=f"{n_customers:,} customers"
    )
with k2:
    st.metric(
        label="🚨 Customers At Risk",
        value=str(len(at_risk)),
        delta=f"{fmt_pct(len(at_risk)/n_customers)} of base",
        delta_color="inverse"
    )
with k3:
    st.metric(
        label="⭐ High Value Customers",
        value=str(len(high_value)),
        delta=f"{fmt_pct(high_value['Predicted_Revenue'].sum()/total_rev if total_rev>0 else 0)} of revenue"
    )
with k4:
    st.metric(
        label="📉 Avg Churn Risk",
        value=fmt_pct(avg_churn),
        delta_color="inverse"
    )

st.markdown("---")


# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("Customer Segments")

col_l, col_r = st.columns([1, 2])

color_map = {
    'High Value': '#10b981',
    'Mid Value':  '#f59e0b',
    'Low Value':  '#6b7280',
    'At Risk':    '#ef4444',
}

with col_l:
    seg_counts = results['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig_pie = px.pie(
        seg_counts, names='Segment', values='Count',
        color='Segment', color_discrete_map=color_map,
        hole=0.55,
    )
    fig_pie.update_traces(textinfo='percent+label', textfont_size=12)
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    seg_rev = results.groupby('Segment')['Predicted_Revenue'].sum().reset_index()
    seg_rev = seg_rev.sort_values('Predicted_Revenue', ascending=True)
    fig_bar = px.bar(
        seg_rev, x='Predicted_Revenue', y='Segment', orientation='h',
        color='Segment', color_discrete_map=color_map,
        labels={'Predicted_Revenue': f'Forecast Revenue ({currency_symbol})', 'Segment': ''},
    )
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat=','),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")


# ── Churn histogram ───────────────────────────────────────────────────────────
st.subheader("Churn Risk Distribution")
st.caption("Shows how spread out the churn risk is across your customer base. Anything above 0.65 is flagged as At Risk.")

fig_hist = px.histogram(
    results, x='Churn_Risk', nbins=30,
    color_discrete_sequence=['#3b82f6'],
    labels={'Churn_Risk': 'Churn Risk Score (0 = safe · 1 = leaving soon)'},
)
fig_hist.add_vline(
    x=0.65, line_dash="dash", line_color="#ef4444",
    annotation_text="At-Risk threshold", annotation_position="top right"
)
fig_hist.update_layout(
    height=220,
    margin=dict(t=10, b=10, l=10, r=10),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    yaxis_title='Number of customers',
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")


# ── Action tips ───────────────────────────────────────────────────────────────
st.subheader("What to do right now")

a1, a2, a3 = st.columns(3)

with a1:
    rev_at_risk = at_risk['Predicted_Revenue'].sum()
    st.error(
        f"**🚨 Rescue {len(at_risk)} at-risk customers**\n\n"
        f"They represent {fmt_currency(rev_at_risk, currency_symbol)} in forecast revenue. "
        f"Send a personalised win-back offer within 7 days."
    )
with a2:
    st.warning(
        f"**📈 Upsell {len(mid_value)} mid-value customers**\n\n"
        f"A 10% increase in their spend adds "
        f"{fmt_currency(mid_value['Predicted_Revenue'].sum()*0.1, currency_symbol)} to your forecast."
    )
with a3:
    st.success(
        f"**⭐ Reward {len(high_value)} VIP customers**\n\n"
        f"They drive {fmt_pct(high_value['Predicted_Revenue'].sum()/total_rev if total_rev>0 else 0)} "
        f"of forecast revenue. Early access + loyalty perks keep them loyal."
    )

st.markdown("---")


# ── Customer table ────────────────────────────────────────────────────────────
st.subheader("Customer List")

tab_all, tab_risk, tab_vip, tab_low = st.tabs([
    f"All ({n_customers})",
    f"At Risk ({len(at_risk)})",
    f"High Value ({len(high_value)})",
    f"Low Value ({len(results[results['Segment']=='Low Value'])})"
])

def show_table(subset):
    display = subset[[
        'CustomerID','Predicted_Revenue','Churn_Risk','Segment',
        'Frequency','Recency','Monetary'
    ]].copy()
    display.columns = [
        'Customer ID', f'Forecast Rev ({currency_symbol})',
        'Churn Risk %', 'Segment',
        'Total Orders', 'Days Since Last Order', f'Total Spent ({currency_symbol})'
    ]
    display[f'Forecast Rev ({currency_symbol})']  = display[f'Forecast Rev ({currency_symbol})'].round(0).astype(int)
    display[f'Total Spent ({currency_symbol})']   = display[f'Total Spent ({currency_symbol})'].round(0).astype(int)
    display['Churn Risk %'] = (display['Churn Risk %'] * 100).round(1)
    display = display.sort_values(f'Forecast Rev ({currency_symbol})', ascending=False)
    st.dataframe(display, use_container_width=True, height=360)

with tab_all:  show_table(results)
with tab_risk: show_table(at_risk)
with tab_vip:  show_table(high_value)
with tab_low:  show_table(results[results['Segment'] == 'Low Value'])

st.markdown("---")


# ── Exports ───────────────────────────────────────────────────────────────────
st.subheader("Export for CRM / Email Tool")
st.caption("Download these CSVs and upload directly to Klaviyo, Mailchimp, or any CRM.")

export = results[['CustomerID','Predicted_Revenue','Churn_Risk','Segment',
                   'Frequency','Recency','Monetary','Avg_Order_Value']].copy()
export.columns = [
    'CustomerID', f'Forecast_Revenue_{currency_symbol}', 'Churn_Risk_Score',
    'Segment', 'Order_Count', 'Days_Since_Last_Order',
    f'Total_Spent_{currency_symbol}', f'Avg_Order_{currency_symbol}'
]
export[f'Forecast_Revenue_{currency_symbol}'] = export[f'Forecast_Revenue_{currency_symbol}'].round(2)
export['Churn_Risk_Score'] = export['Churn_Risk_Score'].round(4)
export = export.sort_values('Churn_Risk_Score', ascending=False)

e1, e2, e3 = st.columns(3)
with e1:
    st.download_button(
        "⬇️ Full customer list",
        data=export.to_csv(index=False).encode(),
        file_name="all_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e2:
    st.download_button(
        "🚨 At-risk customers only",
        data=export[export['Segment']=='At Risk'].to_csv(index=False).encode(),
        file_name="at_risk_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e3:
    st.download_button(
        "⭐ VIP customers only",
        data=export[export['Segment']=='High Value'].to_csv(index=False).encode(),
        file_name="vip_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )