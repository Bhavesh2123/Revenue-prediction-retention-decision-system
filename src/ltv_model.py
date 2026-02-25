from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(model_df):
    X= model_df.drop(['CustomerID', 'Future_6M_Revenue'], axis=1)
    y = model_df['Future_6M_Revenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model= XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8)
    model.fit(X_train,y_train)
    preds= model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    return model