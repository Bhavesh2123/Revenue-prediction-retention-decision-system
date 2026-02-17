# 🛒 E-commerce Customer Segmentation & Lifetime Value (LTV) Prediction

## 📌 Overview
This project analyses real-world e-commerce transactional data to:
- **Understand customer behaviour** through Exploratory Data Analysis (EDA)
- **Segment customers** using unsupervised learning (RFM + clustering)
- **Predict Customer Lifetime Value (LTV)** using supervised machine-learning models  

The aim is to help businesses identify high-value customers, tailor marketing strategies, and optimise customer retention.

---

## 🚀 Features
- **Data Cleaning & Preprocessing**  
  - Removed cancelled/refunded orders  
  - Handled missing Customer IDs  
  - Created key features: `TotalPurchase = Quantity × UnitPrice`  
  - Converted date columns to proper `datetime` format

- **Exploratory Data Analysis (EDA)**  
  - Time-based purchase trends  
  - Revenue & customer growth analysis  
  - Outlier detection

- **Customer Segmentation**  
  - RFM (Recency, Frequency, Monetary) scoring  
  - K-Means clustering to group customers into segments such as:
    * 🏆 **Champions**
    * 💡 **Potential Loyalists**
    * 💤 **At-Risk**
    * ❗ **Hibernating**

- **Lifetime Value (LTV) Prediction**  
  - Engineered features from historical transactions  
  - Built regression models (e.g. Linear Regression / Random Forest)  
  - Evaluated using RMSE & R²

---

## 🛠️ Tech Stack
- **Language**: Python 3.x  
- **Libraries**:
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – clustering & regression models
  - `lifetimes` – customer lifetime metrics (if used)
  - `jupyter` – interactive notebooks

---



