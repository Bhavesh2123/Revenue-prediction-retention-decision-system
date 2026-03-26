import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    initial_rows = len(df)
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"[load_data] Dropped {dropped} rows with missing CustomerID or InvoiceDate")
    return df


# FIX: removed duplicate definition — only one add_total_price now
def add_total_price(df):
    df = df.copy()
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
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
    df = df.sort_values('InvoiceDate')
    past_data = df[df['InvoiceDate'] <= cutoff_date]
    future_data = df[
        (df['InvoiceDate'] > cutoff_date) &
        (df['InvoiceDate'] <= prediction_end)
    ]
    if past_data.empty:
        raise ValueError("past_data is empty — check your cutoff_date.")
    if future_data.empty:
        raise ValueError("future_data is empty — check your prediction_end date.")
    return past_data, future_data


# FIX: added helper to auto-detect sensible cutoff dates from any dataset
def auto_detect_dates(df, obs_months=6):
    """
    Automatically derives cutoff and prediction_end from the dataset's
    own date range. Removes the hardcoded 2011 dates so any D2C client's
    data works out of the box.

    Args:
        df (pd.DataFrame): Must have InvoiceDate column.
        obs_months (int): How many months to use as the observation window.

    Returns:
        tuple: (cutoff_date, prediction_end) as pd.Timestamps.
    """
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    total_days = (max_date - min_date).days

    if total_days < 60:
        raise ValueError(
            f"Dataset only spans {total_days} days — need at least 60 days of history."
        )

    cutoff_date = min_date + pd.DateOffset(months=obs_months)
    if cutoff_date >= max_date:
        # fall back to 70/30 split
        cutoff_date = min_date + pd.Timedelta(days=int(total_days * 0.7))

    prediction_end = max_date
    print(f"[auto_detect_dates] Observation window: {min_date.date()} → {cutoff_date.date()}")
    print(f"[auto_detect_dates] Prediction window:  {cutoff_date.date()} → {prediction_end.date()}")
    return cutoff_date, prediction_end
