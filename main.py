from scripts.train_revenue_model import run_revenue_training
from scripts.train_churn_model import run_churn_training


if __name__ == "__main__":
    run_revenue_training()
    run_churn_training()

    print("All models trained successfully.")