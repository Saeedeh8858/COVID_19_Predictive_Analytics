# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df= pd.read_csv('https://raw.githubusercontent.com/Saeedeh8858/Covid-19/refs/heads/main/covid19.csv')
print("✅ Dataset Overview")
print(f"Number of Records: {df.shape[0]}")
print(f"Number of Features: {df.shape[1]}")
print(f"Features: {', '.join(df.columns)}")

print(df.shape)
print(df.info())

from sklearn.preprocessing import LabelEncoder
import pandas as pd

def clean_covid_data(df, missing_threshold=0.5, outlier_thresholds=None, save_path="cleaned_covid_data.csv"):
    df['Vaccine_Type'] = df['Vaccine_Type'].fillna('Unknown')
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    valid_date_cols = []
    for col in date_cols:
        if col in df.columns:
            converted = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
                valid_date_cols.append(col)
            else:
                df = df.drop(columns=col)
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > missing_threshold].index
    df = df.drop(columns=cols_to_drop)
    df = df.drop_duplicates()
    if outlier_thresholds:
        for col, (min_val, max_val) in outlier_thresholds.items():
            if col in df.columns:
                before = df.shape[0]
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
                after = df.shape[0]
                print(f"{col}: removed {before - after} outliers")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object' or col in valid_date_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    df.to_csv(save_path, index=False)
    print(f"\n✅ Cleaned data saved to: {save_path}")
    print("\n✅ Data cleaned. Final dataframe info:")
    print(df.info())
    return df, label_encoders

outlier_limits = {'BMI': (15, 40), 'Age': (18, 90)}
df_cleaned, encoders = clean_covid_data(df, missing_threshold=0.5, outlier_thresholds=outlier_limits)

from google.colab import drive
drive.mount('/content/drive')

df_cleaned.describe()

import pandas as pd

df = df_cleaned

print("=== Numeric Features Summary ===")
print(df[['Age', 'BMI', 'Doses_Received']].describe())

categorical_cols = ['Gender', 'Region', 'Preexisting_Condition', 'COVID_Strain', 'Symptoms',
                    'Severity', 'Hospitalized', 'ICU_Admission', 'Ventilator_Support',
                    'Recovered', 'Reinfection', 'Vaccination_Status', 'Occupation', 'Smoking_Status']

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(10, 4))
sns.histplot(df_cleaned['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df_cleaned['BMI'], bins=30, kde=True, color='lightgreen')
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df_cleaned, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Severity', data=df_cleaned, palette='muted')
plt.title('Severity Levels')
plt.xlabel('Severity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Hospitalized', data=df_cleaned, palette='bright')
plt.title('Hospitalized vs. Non-Hospitalized')
plt.xlabel('Hospitalized (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Vaccination_Status', data=df_cleaned, palette='cool')
plt.title('Vaccination Status')
plt.xlabel('Vaccination Status (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.boxplot(x='Hospitalized', y='BMI', data=df)
plt.title('BMI by Hospitalization Status')
plt.xlabel('Hospitalized (0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Severity', y='Age', data=df)
plt.title('Age by Severity Level')
plt.xlabel('Severity (0: Mild → 3: Critical)')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='ICU_Admission', y='BMI', data=df)
plt.title('BMI by ICU Admission')
plt.xlabel('ICU Admission (0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Ventilator_Support', y='BMI', data=df)
plt.title('BMI by Ventilator Support')
plt.xlabel('Ventilator Support (0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()

!pip install imblearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

X = df_cleaned[['Vaccination_Status', 'Vaccine_Type', 'Gender', 'Preexisting_Condition',
                'Region', 'COVID_Strain', 'Severity', 'Smoking_Status', 'Doses_Received', 'Age']]
y = df_cleaned['Reinfection']

categorical_features = ['Vaccination_Status', 'Vaccine_Type', 'Gender', 'Preexisting_Condition',
                        'Region', 'COVID_Strain', 'Severity', 'Smoking_Status']
numeric_features = ['Doses_Received', 'Age']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

X = df_cleaned[['Region','Age','BMI','Gender']]
y = df_cleaned['Reinfection']

X_encoded = X.apply(LabelEncoder().fit_transform)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

X = df_cleaned[['Doses_Received', 'Smoking_Status', 'Symptoms', 'COVID_Strain', 'Preexisting_Condition']]
y = df_cleaned['ICU_Admission']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

target ='Hospitalized'
features = [
    'Preexisting_Condition','COVID_Strain','Smoking_Status','Occupation','Vaccine_Type','Vaccination_Status','Severity'
    ,'ICU_Admission','Ventilator_Support'
]

X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pipeline_rf = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline_rf.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)

print("Random Forest Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, digits=3))

print("Accuracy:", accuracy_score(y_test, y_pred_rf))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

df_cleaned['Recovery_Days'] = (df_cleaned['Date_of_Recovery'] - df_cleaned['Date_of_Infection']).dt.days
df_cleaned['Fast_Recovery'] = (df_cleaned['Recovery_Days'] < 14).astype(int)

X = df_cleaned[['Region','Age','BMI','Gender']]
y = df_cleaned['Fast_Recovery']

X_encoded = X.apply(LabelEncoder().fit_transform)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

X = df_cleaned[['BMI','Region','Age','Vaccination_Status','Doses_Received','Reinfection','Severity','Smoking_Status','COVID_Strain','Symptoms']]
y = df_cleaned['Vaccine_Type']

X_encoded = X.apply(LabelEncoder().fit_transform)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

!pip install catboost

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE

target = 'Ventilator_Support'
features = [
    'Age', 'Gender', 'BMI', 'Preexisting_Condition', 'Severity', 'ICU_Admission',
    'Hospitalized', 'Symptoms', 'Smoking_Status', 'Vaccination_Status', 'Doses_Received'
]

present_features = [col for col in features if col in df_cleaned.columns]
if len(present_features) != len(features):
    print(f"Warning: Some features not found in df_cleaned: {list(set(features) - set(present_features))}")
X = df_cleaned[present_features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
ratio_neg_to_pos = neg_count / pos_count

print(f"Ratio of negative to positive samples in training data: {ratio_neg_to_pos:.2f}")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = XGBClassifier(random_state=42, scale_pos_pos_weight=ratio_neg_to_pos, use_label_encoder=False, eval_metric='logloss')

print("Training XGBoost ...")
model.fit(X_train_res, y_train_res)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

print("\nEvaluation for Ventilator_Support Prediction (XGBoost):")
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = df_cleaned[['BMI','Age','Symptoms','COVID_Strain','Preexisting_Condition']]
y = df_cleaned['ICU_Admission']

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

X = df_cleaned[['Region','Age','Gender','Occupation','BMI','Symptoms','Severity','Preexisting_Condition','Hospitalized','Reinfection','Smoking_Status']]
y = df_cleaned['Vaccine_Type']

X_encoded = X.apply(LabelEncoder().fit_transform)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

def categorize_age(age):
    if age < 20:
        return 'Youth'
    elif 30 <= age < 50:
        return 'Adult'
    else:
        return 'Elderly'

df_cleaned['Age_Group'] = df_cleaned['Age'].apply(categorize_age)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

features = ['Preexisting_Condition', 'Severity', 'ICU_Admission', 'Region','COVID_Strain','Symptoms','Hospitalized','ICU_Admission',
            'Reinfection', 'Doses_Received', 'Occupation', 'Smoking_Status',
            'Vaccine_Type', 'Vaccination_Status']

X = df_cleaned[features]
y = df_cleaned['Age_Group']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

!pip install streamlit

import streamlit as st
import numpy as np
import pandas as pd

def predict_q1(inputs):
    return "Low Risk of Reinfection"

def predict_q2(inputs):
    return "High Risk of Reinfection"

def predict_q3(inputs):
    return "ICU Admission Likely"

def predict_q4(inputs):
    return "No Need for Ventilator"

def predict_q5(inputs):
    return "Fast Recovery Predicted"

def predict_q6(inputs):
    return "High Risk of Hospitalization"

def predict_q7(inputs):
    return "Ventilator Required"

def predict_q8(inputs):
    return "ICU Admission Not Likely"

def predict_q9(inputs):
    return "Cluster 1 - Moderate Severity"

def predict_q10(inputs):
    return "Cluster 3 - High Demand Region"

questions = {
    "Q1 - Predict Reinfection of COVID-19 Likelihood based on vaccination": predict_q1,
    "Q2 - Identify Individuals at Reinfection Risk based on demographics": predict_q2,
    "Q3 - ICU Admission Prediction": predict_q3,
    "Q4 - Hospitaazation Prediction": predict_q4,
    "Q5 - Fast Recovery Prediction": predict_q5,
    "Q6 - Vaccine_Type Prediction": predict_q6,
    "Q7 - Predict Need for Ventilator": predict_q7,
    "Q8 - Vaccination and ICU Risk": predict_q8,
    "Q9 - Clustering Patients by Type of Vaccinate": predict_q9,
    "Q10 - Clustering Regions by Demand": predict_q10,
}

st.title("COVID-19 Predictive Dashboard")
st.sidebar.header("Choose a Question")

selected_question = st.sidebar.selectbox("Select a question to analyze:", list(questions.keys()))

st.subheader(selected_question)

with st.form(key='input_form'):
    age = st.slider("Age", 18, 90, 50)
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    vaccinated = st.selectbox("Vaccinated", ["Yes", "No"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    smoking = st.selectbox("Smoker", ["Yes", "No"])
    symptoms = st.multiselect("Symptoms", ["Fever", "Cough", "Fatigue", "Breathing Difficulty"])
    preexisting = st.multiselect("Preexisting Conditions", ["Diabetes", "Heart Disease", "Hypertension"])
    covid_strain = st.selectbox("COVID Strain", ["Alpha", "Delta", "Omicron"])

    submit = st.form_submit_button("Run Prediction")

if submit:
    inputs = {
        "age": age,
        "bmi": bmi,
        "gender": gender,
        "vaccinated": vaccinated,
        "region": region,
        "smoking": smoking,
        "symptoms": symptoms,
        "preexisting": preexisting,
        "covid_strain": covid_strain,
    }

    result = questions[selected_question](inputs)
    st.success(f"Prediction Result: {result}")