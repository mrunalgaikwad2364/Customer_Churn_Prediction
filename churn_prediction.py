import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/Churn_Modelling.csv")

# Show first 5 rows
print(df.head())

# Show basic info
print(df.info())

# Drop unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Check for missing values
print(df.isnull().sum())

# One-hot encode Geography (France, Spain, Germany)
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Label encode Gender (Male/Female → 1/0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

from imblearn.over_sampling import SMOTE

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new counts
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())

# Train model on balanced data
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import shap
import matplotlib.pyplot as plt

plt.switch_backend('tkagg')

# Take 100 test rows and fix column names
sample_data = X_test.iloc[:100]
sample_data = pd.DataFrame(sample_data.values, columns=X_train.columns)

# Make sure all columns are numeric (int or float)
sample_data = sample_data.astype(float)

# SHAP Explainer (XGBoost works well)
explainer = shap.Explainer(model)

# Get SHAP values
shap_values = explainer(sample_data)

# Plot
shap.plots.beeswarm(shap_values)

# Save trained model
import pickle
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved successfully!")


