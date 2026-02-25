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

df= load_data("Dataset/Raw_Data.csv")
df = add_total_price(df)

cutoff_date= pd.Timestamp("2011-06-30")
churn_window_end = pd.Timestamp("2011-09-30")

past_data, future_data= time_based_split(
    df, cutoff_date, pd.Timestamp("2011-12-31")
)

rfm = build_rfm(past_data, cutoff_date)
extra = build_additional_features(past_data, cutoff_date)

features_df = rfm.merge(extra, on="CustomerID", how="left")

active_customers =(future_data[future_data["InvoiceDate"] <= churn_window_end]
                   ['CustomerID'].unique())

features_df['Churn']= (
    ~features_df['CustomerID'].isin(active_customers).astype(int)
)
# 1= churned 0= retained

X= features_df.drop(['CustomerID','Churn'],axis=1)
y= features_df['Churn']
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)
model =XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8)
model.fir(X_train,y_train)

preds= model.predict(X_test)
probs= model.predict_proba(X_test)[:, 1]

print(classification_report(y_test,preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))

joblib.dump(model, "models/churn_model.pkl")
print("Churn model trained successfully.")


