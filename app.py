import streamlit as st
import numpy as np
import joblib

# Page configuration

st.set_page_config(
page_title="Loan Default Prediction",
page_icon="💳",
layout="wide"
)

# Custom CSS for UI

st.markdown("""

<style>
body {
    background: linear-gradient(120deg,#1f4037,#99f2c8);
}

.big-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#ffffff;
}

.sub-text {
    text-align:center;
    color:#eeeeee;
}

.card {
    background-color:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.2);
}
</style>

""", unsafe_allow_html=True)

# Load model

model = joblib.load("loan_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title

st.markdown('<p class="big-title">💳 Loan Default Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Predict whether a customer will default on a loan</p>', unsafe_allow_html=True)

st.write("")

# Input section

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)


# Loan purpose dropdown
purpose = st.selectbox(
    "Loan Purpose",
    [
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business"
    ]
)

col1, col2 = st.columns(2)

with col1:
    credit_policy = st.number_input("Credit Policy (0 or 1)", 0, 1)
    int_rate = st.number_input("Interest Rate")
    installment = st.number_input("Installment")
    log_annual_inc = st.number_input("Log Annual Income")
    dti = st.number_input("Debt to Income Ratio")
    fico = st.number_input("FICO Credit Score")

with col2:
    days_with_cr_line = st.number_input("Days with Credit Line")
    revol_bal = st.number_input("Revolving Balance")
    revol_util = st.number_input("Revolving Utilization")
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months")
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years")
    pub_rec = st.number_input("Public Records")

st.markdown("</div>", unsafe_allow_html=True)


st.write("")


# Prediction button

if st.button("🔍 Predict Loan Risk"):

    # Convert purpose into dummy variables
    purpose_credit_card = 1 if purpose == "credit_card" else 0
    purpose_debt_consolidation = 1 if purpose == "debt_consolidation" else 0
    purpose_educational = 1 if purpose == "educational" else 0
    purpose_home_improvement = 1 if purpose == "home_improvement" else 0
    purpose_major_purchase = 1 if purpose == "major_purchase" else 0
    purpose_small_business = 1 if purpose == "small_business" else 0

    input_data = np.array([[

        credit_policy,
        int_rate,
        installment,
        log_annual_inc,
        dti,
        fico,
        days_with_cr_line,
        revol_bal,
        revol_util,
        inq_last_6mths,
        delinq_2yrs,
        pub_rec,

        purpose_credit_card,
        purpose_debt_consolidation,
        purpose_educational,
        purpose_home_improvement,
        purpose_major_purchase,
        purpose_small_business

    ]])

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)

    st.write("")

    if prediction[0] == 1:
        st.error("⚠ High Risk: Loan may not be fully paid")
    else:
        st.success("✅ Low Risk: Loan likely to be paid")