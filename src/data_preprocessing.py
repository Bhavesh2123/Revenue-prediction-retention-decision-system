import pandas as pd
def load_data(path):
    df=pd.read_csv(path)
    df['InvoiceDate']=pd.to_datetime(df[InvoiceDate])