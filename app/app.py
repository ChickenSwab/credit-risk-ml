import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ─── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('../models/lgbm_best_model.pkl')

model = load_model()

# ─── Header ────────────────────────────────────────────────
st.title("🏦 Credit Risk & Loan Default Predictor")
st.markdown("Enter applicant details below to predict default probability.")
st.divider()

# ─── Input Form ────────────────────────────────────────────
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    amt_income = st.number_input("Annual Income (₹)", 
                                  min_value=10000, max_value=10000000, 
                                  value=200000, step=5000)
    amt_credit = st.number_input("Loan Amount (₹)", 
                                  min_value=10000, max_value=5000000, 
                                  value=500000, step=10000)
    amt_annuity = st.number_input("Monthly EMI (₹)", 
                                   min_value=1000, max_value=500000, 
                                   value=25000, step=1000)

with col2:
    age = st.slider("Age (Years)", min_value=18, max_value=70, value=35)
    years_employed = st.slider("Years Employed", 
                                min_value=0, max_value=40, value=5)
    ext_source_2 = st.slider("Credit Score 1 (0-1)", 
                               min_value=0.0, max_value=1.0, 
                               value=0.5, step=0.01)

with col3:
    ext_source_3 = st.slider("Credit Score 2 (0-1)", 
                               min_value=0.0, max_value=1.0, 
                               value=0.5, step=0.01)
    ext_source_1 = st.slider("Credit Score 3 (0-1)", 
                               min_value=0.0, max_value=1.0, 
                               value=0.5, step=0.01)
    region_rating = st.selectbox("Region Risk Rating", [1, 2, 3], index=1)

st.divider()

# ─── Feature Engineering (mirror Phase 5) ──────────────────
def build_features(income, credit, annuity, age, years_emp,
                   ext1, ext2, ext3, region):

    credit_to_income    = credit / income
    annuity_to_income   = annuity / income
    credit_term         = credit / annuity if annuity > 0 else 0
    goods_to_credit     = 0.9   # default assumption
    employment_to_age   = years_emp / age if age > 0 else 0
    income_per_person   = income / 2
    ext_mean            = np.mean([ext1, ext2, ext3])
    ext_min             = np.min([ext1, ext2, ext3])
    ext_max             = np.max([ext1, ext2, ext3])
    ext_std             = np.std([ext1, ext2, ext3])

    return {
        'AMT_INCOME_TOTAL':          income,
        'AMT_CREDIT':                credit,
        'AMT_ANNUITY':               annuity,
        'DAYS_BIRTH':                -age * 365,
        'DAYS_EMPLOYED':             -years_emp * 365,
        'EXT_SOURCE_1':              ext1,
        'EXT_SOURCE_2':              ext2,
        'EXT_SOURCE_3':              ext3,
        'REGION_RATING_CLIENT':      region,
        'CREDIT_TO_INCOME_RATIO':    credit_to_income,
        'ANNUITY_TO_INCOME_RATIO':   annuity_to_income,
        'CREDIT_TERM':               credit_term,
        'GOODS_TO_CREDIT_RATIO':     goods_to_credit,
        'EMPLOYMENT_TO_AGE_RATIO':   employment_to_age,
        'INCOME_PER_PERSON':         income_per_person,
        'EXT_SOURCE_MEAN':           ext_mean,
        'EXT_SOURCE_MIN':            ext_min,
        'EXT_SOURCE_MAX':            ext_max,
        'EXT_SOURCE_STD':            ext_std,
    }

# ─── Predict Button ────────────────────────────────────────
if st.button("🔍 Predict Default Risk", use_container_width=True):

    features = build_features(
        amt_income, amt_credit, amt_annuity,
        age, years_employed,
        ext_source_1, ext_source_2, ext_source_3,
        region_rating
    )

    # Build full feature vector (fill missing cols with 0)
    X_train_cols = model.feature_name_
    input_df = pd.DataFrame([features])
    input_df = input_df.reindex(columns=X_train_cols, fill_value=0)

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    percent = prob * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    # ─── Risk Category ─────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Default Probability", f"{percent:.1f}%")

    with col_b:
        if prob < 0.3:
            st.success("✅ LOW RISK — Likely to Repay")
        elif prob < 0.5:
            st.warning("⚠️ MEDIUM RISK — Review Required")
        else:
            st.error("🚨 HIGH RISK — Likely to Default")

    # ─── Risk Gauge ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(['Risk'], [prob], color=(
        '#2ecc71' if prob < 0.3 else
        '#f39c12' if prob < 0.5 else
        '#e74c3c'
    ), height=0.4)
    ax.barh(['Risk'], [1 - prob], left=[prob],
            color='#ecf0f1', height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Default Probability')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Risk Gauge')
    st.pyplot(fig)

    # ─── Key Ratios ────────────────────────────────────────
    st.divider()
    st.subheader("📈 Financial Ratios")

    r1, r2, r3 = st.columns(3)
    r1.metric("Credit-to-Income", 
              f"{features['CREDIT_TO_INCOME_RATIO']:.2f}x",
              delta="High Risk" if features['CREDIT_TO_INCOME_RATIO'] > 5 
                    else "Acceptable", 
              delta_color="inverse")
    r2.metric("EMI Burden", 
              f"{features['ANNUITY_TO_INCOME_RATIO']*100:.1f}%",
              delta="High" if features['ANNUITY_TO_INCOME_RATIO'] > 0.5 
                    else "Normal",
              delta_color="inverse")
    r3.metric("Loan Term", 
              f"{features['CREDIT_TERM']:.0f} months")

    # ─── SHAP Explanation ──────────────────────────────────
    st.divider()
    st.subheader("🔎 Why This Prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        ev = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        ev = explainer.expected_value

    # Top 10 contributing features
    shap_series = pd.Series(sv, index=X_train_cols).abs()
    top_features = shap_series.nlargest(10).index.tolist()
    top_vals = pd.Series(sv, index=X_train_cols)[top_features]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_vals]
    ax2.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_title('Feature Impact on Default Probability\n'
                  '(Red = increases risk, Green = decreases risk)')
    ax2.set_xlabel('SHAP Value')
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()
st.caption("Credit Risk ML System | Built with LightGBM + SHAP + Streamlit")