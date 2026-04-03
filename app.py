import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.request
import pickle
import gdown

# =========================
# DOWNLOAD FUNCTION
# =========================
def download_file(file_id, output):
    # Remove corrupted/incomplete file
    if os.path.exists(output):
        if os.path.getsize(output) < 1000000:  # <1MB = corrupted
            os.remove(output)

    # Download if not present
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False, fuzzy=True)


# =========================
# DOWNLOAD ALL FILES
# =========================
download_file("1TieA4CbysErDrv1C5pizz7wf0gFQguFt", "model_reg.pkl")
download_file("13VUiQyVu9D4z6_GFK9-k9XWXxzH0k-pI", "model_clf.pkl")
download_file("1j9s1GXzL8U7-4D6dN5NU65t98p48ew-A", "columns_reg.pkl")
download_file("1ZVepl5Csq932MJNaWNDpQGlj3gmea4tX", "columns_clf.pkl")
download_file("1gwC2LKLDNCfgneCL51zwkCqRqmKZdbd7", "scaler_clf.pkl")
download_file("1iTMdQuV_2BzL0LTq88d21JaHhOCuityO", "label_encoder.pkl")


# =========================
# LOAD MODELS
# =========================
model_reg = pickle.load(open("model_reg.pkl", "rb"))
model_clf = pickle.load(open("model_clf.pkl", "rb"))

columns_reg = pickle.load(open("columns_reg.pkl", "rb"))
columns_clf = pickle.load(open("columns_clf.pkl", "rb"))

scaler_clf = pickle.load(open("scaler_clf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ================================
# TITLE
# ================================
st.title("💼 AI-Powered Credit Risk & EMI Intelligence Platform")

# ================================
# SIDEBAR
# ================================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠Home", "📊Credit Risk Classification", "💰EMI Affordability Analysis"]
)

# ================================
# FEATURE ENGINEERING
# ================================
def feature_engineering(df):

    df["existing_loans"] = df["existing_loans"].map({"Yes": 1, "No": 0})

    df["total_expenses"] = (
        df["monthly_rent"]
        + df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
    )

    df["rent_to_salary_ratio"] = df["monthly_rent"] / (df["monthly_salary"] + 1)
    df["expense_to_income_ratio"] = df["total_expenses"] / (df["monthly_salary"] + 1)

    df["high_financial_stress"] = (df["expense_to_income_ratio"] > 0.6).astype(int)
    df["has_rent"] = (df["monthly_rent"] > 0).astype(int)
    df["no_credit_history"] = (df["credit_score"] < 600).astype(int)

    return df

# ================================
# USER INPUT
# ================================
def get_user_input():
    data = {}

    st.subheader("👤 Personal Info")
    c1, c2 = st.columns(2)
    with c1:
        data["age"] = st.number_input("Age", 18, 70)
        data["marital_status"] = st.selectbox("Marital Status", ["Single", "Married"])
    with c2:
        data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        data["education"] = st.selectbox("Education", ["High School", "Graduate", "Postgraduate", "Professional"])

    st.subheader("💼 Employment Info")
    c3, c4 = st.columns(2)
    with c3:
        data["monthly_salary"] = st.number_input("Monthly Salary", 0)
        data["years_of_employment"] = st.number_input("Years of Employment", 0)
    with c4:
        data["employment_type"] = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        data["company_type"] = st.selectbox("Company Type", ["MNC", "Startup", "Mid-size", "Small", "Large Indian"])

    st.subheader("🏠 Housing & Family")
    c5, c6 = st.columns(2)
    with c5:
        data["house_type"] = st.selectbox("House Type", ["Rented", "Own", "Family"])
        data["family_size"] = st.number_input("Family Size", 1)
    with c6:
        data["monthly_rent"] = st.number_input("Monthly Rent", 0)
        data["dependents"] = st.number_input("Dependents", 0)

    st.subheader("💸 Monthly Expenses")
    c7, c8 = st.columns(2)
    with c7:
        data["school_fees"] = st.number_input("School Fees", 0)
        data["travel_expenses"] = st.number_input("Travel Expenses", 0)
        data["other_monthly_expenses"] = st.number_input("Other Expenses", 0)
    with c8:
        data["college_fees"] = st.number_input("College Fees", 0)
        data["groceries_utilities"] = st.number_input("Groceries & Utilities", 0)

    st.subheader("🏦 Financial Info")
    c9, c10 = st.columns(2)
    with c9:
        data["existing_loans"] = st.selectbox("Existing Loans", ["Yes", "No"])
        data["credit_score"] = st.number_input("Credit Score", 0, 900)
        data["emergency_fund"] = st.number_input("Emergency Fund", 0)
    with c10:
        data["current_emi_amount"] = st.number_input("Current EMI", 0)
        data["bank_balance"] = st.number_input("Bank Balance", 0)

    st.subheader("📋 Loan Details")
    c11, c12 = st.columns(2)
    with c11:
        data["emi_scenario"] = st.selectbox("EMI Type", ["Personal Loan EMI", "E-commerce EMI", "Education EMI", "Vehicle EMI", "Home Appliances EMI"])
        data["requested_amount"] = st.number_input("Requested Amount", 0)
    with c12:
        data["requested_tenure"] = st.number_input("Tenure (months)", 1)

    return pd.DataFrame([data])

# ================================
# HOME PAGE
# ================================
if menu == "🏠Home":

    st.header("📊 Project Overview")
    st.write("""### 🔍 What it does:
- 📊 Predicts *Maximum EMI Capacity* (Regression Model)
- ⚠️ Classifies *Loan Risk Level* (Low / Medium / High)

### ⚙️ How it works:
- Advanced *Feature Engineering*
- Trained using *Random Forest Models*
- Handles real-world financial behavior patterns

### 🚀 Key Highlights:
- Consistent training vs prediction pipeline
- Smart financial indicators (ratios & stress signals)
- Real-time predictions using Streamlit

### 📌 Use Cases:
- Loan eligibility assessment
- Financial risk profiling
- Credit decision support systems
""")

# ================================
# ELIGIBILITY PREDICTION
# ================================
elif menu == "📊Credit Risk Classification":

    st.header("📊Credit Risk Classification")

    data = get_user_input()

    if st.button("Predict Eligibility"):

        df = feature_engineering(data)
       
        # Encoding
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        # Align columns
        missing_cols = set(columns_clf) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        df = df[columns_clf]

        # Scale
        df_scaled = scaler_clf.transform(df)

        # Predict
        pred = model_clf.predict(df_scaled)[0]

        if pred == 0:
            st.success("Eligible ✅")
        elif pred == 1:
            st.warning("Medium Risk ⚠️")
        else:
            st.error("High Risk ❌")

# ================================
# EMI PREDICTION
# ================================
elif menu == "💰EMI Affordability Analysis":

    st.header("💰EMI Affordability Analysis")

    data = get_user_input()

    if st.button("Predict EMI"):

        # -------------------------------
        # 1. Feature Engineering
        # -------------------------------
        df_reg = feature_engineering(data)

        # -------------------------------
        # 2. Encoding (same as training)
        # -------------------------------
        cat_cols = df_reg.select_dtypes(include="object").columns
        df_reg = pd.get_dummies(df_reg, columns=cat_cols, drop_first=False)

        # -------------------------------
        # 3. Align columns properly
        # -------------------------------
        df_reg = df_reg.reindex(columns=columns_reg)
        df_reg = df_reg.fillna(0)
    
        
       
        emi=model_reg.predict(df_reg)[0]

        # -------------------------------
        # 6. Output
        # -------------------------------
        st.success(f"💰 Estimated EMI Capacity: ₹ {round(emi, 2)}")