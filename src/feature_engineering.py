import pandas as pd


def build_rfm(past_data, snapshot_date):
    """
    Builds Recency, Frequency, Monetary features per customer.
    """
    if past_data.empty:
        raise ValueError("[build_rfm] past_data is empty - check your timesplit")

    rfm = past_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm


def build_additional_features(past_data, snapshot_date):
    features = past_data.groupby('CustomerID').agg(
        Avg_Order_Value=('TotalPrice', 'mean'),
        First_Purchase_Date=('InvoiceDate', 'min'),
        Last_Purchase_Date=('InvoiceDate', 'max'),
        Unique_Products=('StockCode', 'nunique'),
        Total_Quantity=('Quantity', 'sum'),
        Num_Invoices=('InvoiceNo', 'nunique'),
    ).reset_index()

    features['Customer_Age'] = (snapshot_date - features['First_Purchase_Date']).dt.days
    features['Time_Since_Last_Purchase'] = (
        snapshot_date - features['Last_Purchase_Date']
    ).dt.days
    features['Purchase_Velocity'] = (
        features['Num_Invoices'] / (features['Customer_Age'] / 30)
    ).replace([float('inf')], 0).fillna(0)

    return features[[
        'CustomerID', 'Avg_Order_Value', 'Customer_Age',
        'Time_Since_Last_Purchase', 'Unique_Products',
        'Total_Quantity', 'Purchase_Velocity'
    ]]


# FIX: renamed from build_Future_Revenue to build_future_revenue (consistent snake_case)
def build_future_revenue(future_data):
    future_revenue = (
        future_data.groupby('CustomerID')['TotalPrice']
        .sum()
        .reset_index()
    )
    future_revenue.columns = ['CustomerID', 'Future_6M_Revenue']
    return future_revenue


def merge_all_features(rfm, extra_features, future_revenue):
    start = len(rfm)
    model_df = rfm.merge(extra_features, on='CustomerID', how='left')
    model_df = model_df.merge(future_revenue, on='CustomerID', how='left')
    model_df['Future_6M_Revenue'] = model_df['Future_6M_Revenue'].fillna(0)

    zero_rev = (model_df['Future_6M_Revenue'] == 0).sum()
    print(f"[merge] {start} customers | {zero_rev} with no future revenue ({zero_rev / start:.1%} churned)")
    return model_df


def build_churn_target(past_data, future_data):
    """
    Labels each customer as churned (1) or retained (0).
    """
    past_customers = set(past_data['CustomerID'].unique())
    future_customers = set(future_data['CustomerID'].unique())

    # FIX: was pd.Dataframe (wrong capitalisation) — correct is pd.DataFrame
    churn_df = pd.DataFrame({'CustomerID': list(past_customers)})
    churn_df['Churned'] = churn_df['CustomerID'].apply(
        lambda x: 0 if x in future_customers else 1
    )
    return churn_df
