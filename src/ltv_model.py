import pandas as pd
def load_path(path):
    df=pd.read_csv(path, ecofing='latin-1')
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
    