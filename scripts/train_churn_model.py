import json
import pandas as pd
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
df = load_data(BASE_DIR / "Dataset" / "Raw_Data.csv")

from src.data_preprocessing import(
    load_data,add_total_price,time_based_split
)

from src.feature_engineering import (
    build_additional_features, build_rfm
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def run_churn_training():
    df= load_data("Dataset/Raw_Data.csv")
    df = add_total_price(df)

    cutoff_date= pd.Timestamp("2011-06-30")
    churn_window_end = pd.Timestamp("2011-09-30")
    data_end = pd.Timestamp("2011-12-31")

    past_data, future_data= time_based_split(
        df, cutoff_date, data_end
    )
    active_customers = (future_data[future_data["InvoiceDate"] <= churn_window_end]['CustomerID'].unique())

    rfm = build_rfm(past_data, cutoff_date)
    extra = build_additional_features(past_data, cutoff_date)

    features_df = rfm.merge(extra, on="CustomerID", how="left")

    active_customers =(future_data[future_data["InvoiceDate"] <= churn_window_end]
                    ['CustomerID'].unique())

    features_df['Churn']= (
        ~features_df['CustomerID'].isin(active_customers)).astype(int)
    churn_pct = features_df["Churn"].mean()
    print(f"[churn] Churn rate: {churn_pct:.1%}  "
          f"({features_df['Churn'].sum()} churned / {len(features_df)} total)")

# 1= churned 0= retained

    X= features_df.drop(['CustomerID','Churn'],axis=1)
    y= features_df['Churn']
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    
    scale_pos_weight = round((1 - y_train.mean()) / y_train.mean(), 2)
    print(f"[churn] scale_pos_weight set to {scale_pos_weight}")

    param_distributions = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.2, 0.5],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [1, 1.5, 2.0, 5.0],
    }
    base_model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,  # corrects for imbalance
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc",
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=75,
        scoring="roc_auc",       # right metric for imbalanced churn
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True,
    )
    print("[churn] Starting RandomizedSearchCV. . .")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    preds= best_model.predict(X_test)
    probs= best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print("\\n── Churn Model Performance ────────────────")
    print(classification_report(y_test, preds, target_names=["Retained", "Churned"]))
    print(f"  ROC-AUC:      {auc:.4f}")
    print(f"  CV Best AUC:  {search.best_score_:.4f}")
    print(f"  Best Params:  {search.best_params_}")

    metrics = {
        "roc_auc":      round(auc, 4),
        "cv_best_auc":  round(search.best_score_, 4),
        "churn_rate":   round(churn_pct, 4),
        "best_params":  search.best_params_,
    }
    metrics_path = BASE_DIR / "models" / "churn_model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[churn] Metrics saved to {metrics_path}")
    model_path = BASE_DIR / "models" / "churn_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[churn] Model saved to {model_path}")

    return best_model, metrics
if __name__ == "__main__":
    run_churn_training