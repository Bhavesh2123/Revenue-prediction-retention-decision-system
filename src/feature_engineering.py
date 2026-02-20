def build_rfm(past_data, snapshot_date):
    rfm=past_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x:(snapshot_date- x.max()).days,
        'InvoiceNo':'nunique',
        'TotalPrice':'sum'
    }).reset_index()
    rfm.columns=['CustomerID','Recency','Frequency','Monetary']
    return rfm
def build_additional_features(past_data, snapshot_date):
    features= past_data.groupby('CustomerID').agg({
        'TotalPrice': ['mean'],
        'InvoiceDate': ['min','max']
    })
    features.columns = ['Avg_Order_Value','First_Purchase_Date','Last_Purchase_Date']
    features = features.reset_index()
    features['Customer_Age']=(snapshot_date - features['First_Purchase_Date']).dt.days
    features['Time_Since_Last_Purchase'] = (
        snapshot_date - features['Last_Purchase_Date']
    ).dt.days
    return features[['CustomerID','Avg_Order_Value','Customer_Age','Time_Since_Last_Purchase']]

def build_Future_Revenue(future_data):
    future_revenue=(future_data.groupby('CustomerID')['TotalPrice']
                    .sum().reset_index())
    future_revenue.columns= ['CustomerID','Future_6M_Revenue']
    return future_revenue

def merge_all_features(rfm, extra_features, future_revenue):
    model_df=rfm.merge(extra_features, on='CustomerID', how='left')
    model_df=model_df.merge(future_revenue, on='CustomerID', how='left')

    model_df['Future_6M_Revenue']=(
        model_df['Future_6M_Revenue'].fillna(0)
    )
    return model_df
