import pandas as pd
import joblib

from src.data_preprocessing import(
    load_data,add_total_price,time_based_split
)

from src.feature_engineering import (
    build_additional_features, build_rfm
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df= load_data(Dataset/Raw_Data.csv)
df = add_total_price(df)

cutoff_date= pd.Timestamp("2011-06-30")
churn_window_end = pd.Timestamp("2011-09-30")

past_data, future_data= time_based_split(
    df, cutoff_date, pd.Timestamp("2011-12-31")
)

