# 💼 EMI Risk & Eligibility Prediction System

A Machine Learning-powered web application that predicts:

  - 📊 Maximum EMI Repayment Capacity (Regression)
  - ⚠️ Loan Risk Category (Classification)

   Built using Python, Scikit-learn, XGBoost, MLflow, and Streamlit, this project simulates real-world financial decision-making systems used in banking and fintech.

-

## 🚀 Features

  - End-to-end ML pipeline (EDA → Feature Engineering → Modeling → Deployment)
  - Separate models for:
  - EMI prediction (Regression)
  - Risk classification (Classification)
  - Real-time predictions via Streamlit UI
  - MLflow-based experiment tracking
  - Handles:
    - Missing values
    - Feature alignment
    - Data leakage prevention


## 🧠 Key Concepts Used

  - Feature Engineering:
  - Expense-to-Income Ratio
  - Rent-to-Salary Ratio
  - Financial Stress Indicators
  - One-Hot Encoding
  - Train-Test Split
  - Model Evaluation (R², RMSE, Accuracy, F1-score)
  - MLflow Experiment Tracking
  - Model Serialization (Pickle)

## 🛠️ Tech Stack

  - Python  
  - Pandas / NumPy  
  - Scikit-learn  
  - XGBoost  
  - MLflow  
  - Streamlit 

## 🧪 Model Selection & Experiment Tracking

Multiple models were trained and evaluated using MLflow to select the best-performing ones.

### 🔍 Models Evaluated

Regression:
- Linear Regression
- Ridge & Lasso (Polynomial)
- Random Forest (Baseline & Tuned)
- XGBoost (Baseline)

Classification:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

### 📊 Final Model Selection

Regression Model (EMI Prediction):
  - Selected Model: Random Forest (Tuned)
  - R² Score: ~0.979  
  - RMSE: ~1100  

Why chosen:
  - Best balance between bias & variance  
  - Lower RMSE than other models  
  - Stable performance on unseen data  

---

Classification Model (Risk Prediction):
  - Selected Model: XGBoost Classifier  
  - Accuracy: ~93.7%  
  - F1 Score: ~0.937  

Why chosen:
  - Highest classification performance  
  - Handles complex patterns well  
  - Better generalization  

---

### 📈 Experiment Tracking (MLflow)

MLflow was used to:
  - Track experiments  
  - Compare models  
  - Log metrics and parameters  
  - Save trained models  


👉 Attached the tracking link here:  http://127.0.0.1:5000/#/experiments/885067963678930537?viewStateShareKey=ed7515db6761d944feb97e6905108cca4df71d7b4cfd544b7b9dc7649b9e30a5&compareRunsMode=TABLE

## 📂 Dataset

Due to size constraints, the dataset is hosted on Google Drive:

👉  Dataset:  https://drive.google.com/drive/folders/1EHSvWNWdn7RlPqvCmun2C9aaR07O4ACK?usp=drive_link

## 📂 Project Structure

Emi-Pro/
│── app.py
│── README.md
│── .gitignore
│
├── notebooks/
│   └── emi_prediction.ipynb
│
├── assets/
│   └── mlflow path



## ⚠️ Challenges Faced

  - Feature mismatch between training & Streamlit  
  - Data leakage issues  
  - Feature engineering consistency  
  - Column alignment after encoding  


## 💡 Learnings

  - Training and inference pipelines must match exactly  
  - Debugging ML systems is harder than building them  
  - Feature engineering matters more than model choice  
  - MLflow helps in structured experimentation  
