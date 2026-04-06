"""
shap_explainer.py
==================
SHAP explainability for the churn and revenue models.

Install:
    pip install shap

Usage:
    from src.shap_explainer import build_shap_explainer, explain_customer, global_feature_importance
"""

import numpy as np
import pandas as pd
import shap

# ── Plain-English feature name map ───────────────────────────────────────────
# Technical column name → what a D2C owner understands
FEATURE_LABELS = {
    "Recency":                  "Days since last order",
    "Frequency":                "Number of orders placed",
    "Monetary":                 "Total money spent",
    "Avg_Order_Value":          "Average order size",
    "Customer_Age":             "How long they've been a customer (days)",
    "Time_Since_Last_Purchase": "Days since last purchase",
    "Unique_Products":          "Variety of products bought",
    "Total_Quantity":           "Total items purchased",
    "Purchase_Velocity":        "Orders per month (on average)",
}

# Direction map: does a HIGH value of this feature mean MORE or LESS churn risk?
FEATURE_DIRECTION = {
    "Recency":                  "high_bad",   # long gap = bad
    "Frequency":                "high_good",  # more orders = good
    "Monetary":                 "high_good",
    "Avg_Order_Value":          "high_good",
    "Customer_Age":             "high_good",
    "Time_Since_Last_Purchase": "high_bad",
    "Unique_Products":          "high_good",
    "Total_Quantity":           "high_good",
    "Purchase_Velocity":        "high_good",
}


def build_shap_explainer(model, X_background: pd.DataFrame):
    """
    Creates a SHAP TreeExplainer for the given XGBoost model.

    Args:
        model:         Trained XGBClassifier or XGBRegressor.
        X_background:  A sample of training data (100–500 rows) used as background.
                       Pass features_df[MODEL_FEATURES].sample(min(300, len(features_df))).

    Returns:
        shap.TreeExplainer
    """
    explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")
    return explainer


def explain_customer(explainer, customer_row: pd.Series, model_type: str = "churn"):
    """
    Generates a plain-English explanation for a single customer's prediction.

    Args:
        explainer:      shap.TreeExplainer from build_shap_explainer().
        customer_row:   One row from features_df (with MODEL_FEATURES columns).
        model_type:     "churn" or "revenue"

    Returns:
        dict with keys:
            - base_value:     model's average prediction
            - prediction:     this customer's prediction
            - factors:        list of dicts sorted by impact, each with:
                                feature, label, shap_value, direction, plain_reason
    """
    X = customer_row.values.reshape(1, -1)
    shap_vals = explainer.shap_values(X)

    # For classifiers, shap_values returns list [class0, class1] — we want class1 (churn=1)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = shap_vals.flatten()
    feature_names = customer_row.index.tolist()

    factors = []
    for i, feat in enumerate(feature_names):
        sv = float(shap_vals[i])
        val = float(customer_row.iloc[i])
        label = FEATURE_LABELS.get(feat, feat)
        direction = FEATURE_DIRECTION.get(feat, "high_good")

        # Build a human-readable reason
        if model_type == "churn":
            if sv > 0:   # pushes toward churn
                if direction == "high_bad":
                    plain = f"{label} is high ({val:.0f}) — increasing churn risk"
                else:
                    plain = f"{label} is low ({val:.1f}) — increasing churn risk"
            else:        # pushes away from churn
                if direction == "high_good":
                    plain = f"{label} is strong ({val:.1f}) — reducing churn risk"
                else:
                    plain = f"{label} is low ({val:.0f}) — reducing churn risk"
        else:  # revenue
            if sv > 0:
                plain = f"{label} ({val:.1f}) is boosting predicted revenue"
            else:
                plain = f"{label} ({val:.1f}) is limiting predicted revenue"

        factors.append({
            "feature":      feat,
            "label":        label,
            "value":        val,
            "shap_value":   sv,
            "abs_impact":   abs(sv),
            "direction":    "increases_risk" if sv > 0 else "decreases_risk",
            "plain_reason": plain,
        })

    # Sort by absolute impact, descending
    factors = sorted(factors, key=lambda x: x["abs_impact"], reverse=True)

    return {
        "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value),
        "prediction": float(shap_vals.sum() + (explainer.expected_value[1]
                            if isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value)),
        "factors":    factors,
    }


def global_feature_importance(explainer, X: pd.DataFrame, top_n: int = 9):
    """
    Computes global feature importance using mean |SHAP| across all customers.

    Args:
        explainer:  shap.TreeExplainer
        X:          Full feature matrix (features_df[MODEL_FEATURES])
        top_n:      How many features to return

    Returns:
        pd.DataFrame with columns [feature, label, mean_abs_shap]
        sorted descending by mean_abs_shap
    """
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    feature_names = X.columns.tolist()

    df = pd.DataFrame({
        "feature":       feature_names,
        "label":         [FEATURE_LABELS.get(f, f) for f in feature_names],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    return df


def generate_plain_english_summary(explanation: dict, customer_id: str, churn_prob: float):
    """
    Converts SHAP explanation into a 2–3 sentence plain-English summary
    suitable for showing directly to a D2C business owner.

    Returns:
        str
    """
    top_risk_factors    = [f for f in explanation["factors"] if f["direction"] == "increases_risk"][:2]
    top_protect_factors = [f for f in explanation["factors"] if f["direction"] == "decreases_risk"][:1]

    risk_pct = int(churn_prob * 100)

    if churn_prob > 0.65:
        opener = f"Customer {customer_id} has a **{risk_pct}% chance of churning** — action needed."
    elif churn_prob > 0.4:
        opener = f"Customer {customer_id} is showing **early warning signs** ({risk_pct}% churn risk)."
    else:
        opener = f"Customer {customer_id} looks **healthy** with only {risk_pct}% churn risk."

    reasons = []
    for f in top_risk_factors:
        reasons.append(f["plain_reason"])
    risk_text = " ".join(reasons) if reasons else ""

    protect_text = ""
    if top_protect_factors:
        protect_text = f"However, {top_protect_factors[0]['plain_reason'].lower()}."

    return f"{opener} {risk_text} {protect_text}".strip()