import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import joblib
from src.data_preprocessing import(
    load_data,
    add_total_price,
    time_based_split
)
from src.feature_engineering import(
    build_rfm,
    build_additional_features,
    build_Future_Revenue,
    merge_all_features
)

from src.ltv_model import train_model
#Load data
df=load_data("Dataset/Raw_Data.csv")
df= add_total_price(df)

cutoff_date=pd.Timestamp("2011-06-30")
prediction_end= pd.Timestamp("2011-12-31")

#Time Splitted
past_data, future_data= time_based_split(df, cutoff_date, prediction_end)

rfm= build_rfm(past_data, cutoff_date)
extra_features = build_additional_features(past_data, cutoff_date)
future_revenue= build_Future_Revenue(future_data)

model_df= merge_all_features(rfm, extra_features, future_revenue)

model= train_model(model_df)
joblib.dump(model, "scripts/revenue_model.pkl")
print("Training Complete.")


