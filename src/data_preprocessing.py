import pandas as pd
def load_data(path):
    df=pd.read_csv(path)
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
    return df
def add_total_price(df):
    df['TotalPrice']=df['Quantity']*df['UnitPrice']
    return df
def time_based_split(df, cutoff_date, prediction_end):
    df=df.sort_values('InvoiceDate')
    past_data = df[df['InvoiceDate']<=cutoff_date]
    future_data= df[
        (df['InvoiceDate']> cutoff_date)&
        (df['InvoiceDate']<= prediction_end)
    ]
    return past_data, future_data