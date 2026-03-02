import pandas as pd
def load_data(path):
    df=pd.read_csv(path)
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
    initial_rows= len(df)
    df=df.dropna(subset=['CustomerID', 'InvoiveDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    dropped= initial_rows - len(df)
    if dropped > 0:
        print(f"[load_data] Dropped {dropped} rows with missing CustomerID or InvoiceDate")
    return df
def add_total_price(df):
    df=df[df['Quantity']> 0]
    df=df[df['UnitPrice']> 0]
    df['TotalPrice']=df['Quantity']*df['UnitPrice']
    return df
def time_based_split(df, cutoff_date, prediction_end):
    """
    Splits transaction data into past (training) and future (label) windows.

    Args:
        df (pd.DataFrame): Full transaction dataframe with InvoiceDate column.
        cutoff_date (pd.Timestamp): End of the observation window.
        prediction_end (pd.Timestamp): End of the prediction window.

    Returns:
        tuple: (past_data, future_data) as DataFrames.
    """
    df=df.sort_values('InvoiceDate')
    past_data = df[df['InvoiceDate']<=cutoff_date]
    future_data= df[
        (df['InvoiceDate']> cutoff_date)&
        (df['InvoiceDate']<= prediction_end)
    ]
    if past_data.empty:
        raise ValueError("past_data is empty — check your cutoff_date.")
    if future_data.empty:
        raise ValueError("future_data is empty — check your prediction_end date.")
    return past_data, future_data
def add_total_price(df):
    df=df.copy()
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df