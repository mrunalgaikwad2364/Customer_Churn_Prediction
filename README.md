# 💼 Customer Churn Prediction with Explainability & Streamlit Web App

## 📊 Project Overview

This project predicts customer churn for a bank using machine learning. 
It includes:

✅ **XGBoost Classification Model**  
✅ **SMOTE** for handling imbalanced data  
✅ **SHAP Explainability** to understand predictions  
✅ **Streamlit Web App** for interactive prediction  
✅ **Retention Suggestions** to reduce churn  
✅ **Downloadable Prediction Report**

---

## 🚀 Features

- Predict if a customer is likely to churn or stay  
- SHAP visualizations to explain the prediction  
- Retention suggestions based on customer profile  
- Clean, interactive web interface  
- Ability to download predictions as a CSV report  

---

## 🛠️ Tech Stack

- Python  
- XGBoost  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- SHAP  
- Streamlit  
- Pandas, Matplotlib  

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
├── app.py                # Streamlit web app
├── churn_prediction.py   # Model training & SHAP generation
├── xgb_model.pkl         # Saved trained model
├── Churn_Modelling.csv   # Dataset
├── prediction_history.csv # Prediction history (auto-generated)
└── README.md              # Project documentation
```

---

## 📦 Setup & Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

(*Generate requirements with `pip freeze > requirements.txt` if needed.*)

3. **Run the app**
```bash
streamlit run app.py
```

---

## 🎯 Future Enhancements 

- Deploy app on Streamlit Cloud  
- Add probability gauge chart  
- Real-time history table in the app  
- User authentication  

---

## 🙌 Acknowledgements

- Inspired by real-world churn prediction use-cases  
- Dataset provided by CodSoft Internship Program  
- SHAP explainability 

---

## 📬 Contact

Feel free to connect:  
[LinkedIn](https://www.linkedin.com/in/mrunal-gaikwad-328273300) | [GitHub](https://github.com/mrunalgaikwad2364/Customer_Churn_Prediction)  
    
