"""
Run this once to fix the saved model files.
They were saved as (model, metrics) tuples instead of just the model.

Usage:
    python fix_models.py
"""
import joblib
from pathlib import Path

models_dir = Path(__file__).resolve().parent / "models"

for name in ["revenue_model.pkl", "churn_model.pkl"]:
    path = models_dir / name
    if not path.exists():
        print(f"[skip] {name} not found")
        continue

    obj = joblib.load(path)

    if isinstance(obj, tuple):
        model = obj[0]
        joblib.dump(model, path)
        print(f"[fixed] {name} — extracted model from tuple and re-saved")
    else:
        print(f"[ok]    {name} — already correct, no change needed")

print("\nDone. Now run: streamlit run Dashboard.py")