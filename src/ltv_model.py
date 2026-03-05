import json
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
def train_model(model_df):
    """
    Trains an XGBRegressor to predict Future_6M_Revenue using RandomizedSearchCV.

    Args:
        model_df (pd.DataFrame): Merged feature DataFrame with CustomerID
                                 and Future_6M_Revenue columns.

    Returns:
        tuple: (best_estimator, metrics_dict)
    """
    X= model_df.drop(['CustomerID', 'Future_6M_Revenue'], axis=1)
    y = model_df['Future_6M_Revenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Hyper parameter search space
    param_distributions = {
        "n_estimators":[100, 200, 3000, 500],
        "max_depth":[3, 4, 5, 6, 8],
        "learning_rate":[0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample":[0.6, 0.7, 0.8, 0.9, 1.0],
        "comsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":[1, 3, 5, 7],
        "gamma":[0, 0.1, 0.2, 0.5],
        "reg_alpha":[0, 0.01, 0.1, 1.0], #L1 regularixation
        "reg_lambda":[1, 1.5, 2.0, 5.0], #L2 regularization
    }
    base_model= XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    search= RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iters=75,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )
    print("[ltv_mode] Starting RandomizedSearchCV - this may takes a few minutes. . .")
    search.fit(X_train, y_train)

    best_model= search.best_estimator_
    preds = best_model.predict(X_test)

    metrics = {
        "MAE":         round(mean_absolute_error(y_test, preds), 4),
        "RMSE":        round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "R2":          round(r2_score(y_test, preds), 4),
        "best_params": search.best_params_,
        "cv_best_rmse": round(-search.best_score_, 4),
    }
    metrics_path= BASE_DIR / "models" / "revenue_model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        print(f"\\n[ltv_model] Metrics saved to {metrics_path}")

    # feature importance
    importance = dict(zip(X.columns, best_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\\n-- Top 10 Features ----------")
    for feat, score in top_features:
        print(f"  {feat:<30}{score:.4f}")
    return best_model, metrics
