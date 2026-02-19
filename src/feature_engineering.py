def build_rfm(past_data, snapshot_date):
    rfm=past_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x:(snapshot_date- x.max()).days,
        'InvoiceNo':'nunique'
        'TotalPrice':'sum'
    }).reset_index()
    rfm.columns=['CustomerID','Recency','Frequency','Monetary']
    return rfm
def build_additional_features