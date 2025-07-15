import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os

# Load trained model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Page Config
st.set_page_config(page_title="Customer Churn Prediction System", layout="wide", initial_sidebar_state="expanded")

# Sidebar - Customer Info
st.sidebar.header("🧾 Customer Information")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.checkbox("Has Credit Card", True)
is_active = st.sidebar.checkbox("Active Member", True)
salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)
country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])

# Prepare input for model
gender_val = 1 if gender == "Male" else 0
geo_germany = 1 if country == "Germany" else 0
geo_spain = 1 if country == "Spain" else 0
has_card_val = 1 if has_card else 0
is_active_val = 1 if is_active else 0

input_data = np.array([[credit_score, gender_val, age, tenure, balance, products,
                        has_card_val, is_active_val, salary, geo_germany, geo_spain]])
input_df = pd.DataFrame(input_data, columns=[
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
])

# Main Title
st.title("🏦 Customer Churn Prediction System")
st.markdown(" 👋 Welcome!")
st.markdown("Fill in the customer details on the left panel and click the button below to check if the customer is likely to churn.")

# Predict Button
if st.sidebar.button("🚀 Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    if prob < 20:
        st.success(f"✅ LOW CHURN RISK - Probability: {prob:.1f}%")
    else:
        st.error(f"⚠️ HIGH CHURN RISK - Probability: {prob:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Profile Comparison")
        averages = {
            'CreditScore': 650,
            'Age': 40,
            'Tenure': 5,
            'NumOfProducts': 2,
            'Balance': 75000,
            'EstimatedSalary': 100000
        }

        profile_df = pd.DataFrame({
            'Metric': ["Credit Score", "Age", "Tenure", "Products", "Balance ($)", "Salary ($)"],
            'Customer': [credit_score, age, tenure, products, balance, salary],
            'Average': [averages['CreditScore'], averages['Age'], averages['Tenure'],
                        averages['NumOfProducts'], averages['Balance'], averages['EstimatedSalary']]
        })

        chart_df = profile_df.melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig_bar = go.Figure()

        for t in chart_df['Type'].unique():
            df_filtered = chart_df[chart_df['Type'] == t]
            fig_bar.add_trace(go.Bar(x=df_filtered['Metric'], y=df_filtered['Value'], name=t))

        fig_bar.update_layout(barmode='group', height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🛡️ Risk Factors Analysis")
        risk_factors = []

        if age > 50:
            risk_factors.append("Age above 50")
        if not is_active:
            risk_factors.append("Inactive Member")
        if products == 1:
            risk_factors.append("Only one product")

        if risk_factors:
            st.error(f"Identified Risk Factors: {', '.join(risk_factors)}")
        else:
            st.success("No major risk factors identified.")

        st.subheader("💡 Recommendations")
        if risk_factors:
            if "Age above 50" in risk_factors:
                st.write("- Offer senior benefits or personalized retirement plans.")
            if "Inactive Member" in risk_factors:
                st.write("- Engage customer with loyalty rewards or better services.")
            if "Only one product" in risk_factors:
                st.write("- Cross-sell more banking products like loans or insurance.")
        else:
            st.write("- Continue regular engagement with the customer")
            st.write("- Offer product upgrades")
            st.write("- Maintain service quality")
            st.write("- Monitor for any changes")

    st.markdown("---")

    st.subheader("📊 Churn Probability Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Churn Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 20], 'color': "lightgreen"},
                   {'range': [20, 50], 'color': "yellow"},
                   {'range': [50, 100], 'color': "red"}]}
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("🔬 SHAP Explainability")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    plt.figure(figsize=(6, 2))
    shap.plots.force(shap_values[0], matplotlib=True, show=False)
    st.pyplot(plt.gcf())

    st.subheader("📅 Download Prediction Report")
    geo = country
    report_data = {
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Geography": geo,
        "Prediction": "Churn" if prediction else "Stay",
        "Probability": round(prob if prediction else 100 - prob, 2)
    }

    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Report as CSV",
        data=csv,
        file_name='churn_prediction_report.csv',
        mime='text/csv'
    )

    # Save to history
    try:
        existing = pd.read_csv("prediction_history.csv")
        updated = pd.concat([existing, report_df], ignore_index=True)
    except FileNotFoundError:
        updated = report_df

    updated.to_csv("prediction_history.csv", index=False)
    st.success("📄 Prediction saved to history!")

# View History Button
st.sidebar.markdown("---")
if st.sidebar.button("View Prediction History"):
    if os.path.exists("prediction_history.csv"):
        st.subheader("📂 Prediction History")
        history = pd.read_csv("prediction_history.csv")

        # Clean history
        history.dropna(inplace=True)
        expected_cols = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                         'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                         'Geography', 'Prediction', 'Probability']
        history = history[[col for col in expected_cols if col in history.columns]]

        st.dataframe(history)
    else:
        st.info("No prediction history available yet.")
