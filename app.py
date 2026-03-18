# Creation of Calculator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

X = heart_disease.data.features.copy()

# Fill missing 'ca' values with the mode
X['ca'] = X['ca'].fillna(X['ca'].mode()[0])
# Fill missing 'thal' values with the mode
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

y_binary = y['num'].apply(lambda x: 1 if x > 0 else 0)
df = pd.concat([X, y_binary], axis = 1)

df.head()

#averages
avg_age = round(df['age'].mean())
avg_restingbp = round(df['trestbps'].mean())
avg_chols = round(df['chol'].mean())
avg_hr = round(df['thalach'].mean())

# Split into training and testing sets (Binary: 0 = Healthy, 1 = Sick)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size = 0.2, random_state = 42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Multiple Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Convert the scaled array back into a Dataframe with original names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X.columns)

# Add a constant (intercept)
X_train_const = sm.add_constant(X_train_scaled_df)


# Takes a dictionary of patient data and return the probability of heart disease.
def heart_disease_risk_calculator(patient_data):
    # Convert patient dict to a 2D array 
    patient_df = pd.DataFrame([patient_data])

    # Scaling the new data 
    patient_scaled = scaler.transform(patient_df)

    # Predicting the probablity, will return [prob_healthy, prob_sick]
    probability = model.predict_proba(patient_scaled)[0][1]

    return probability

# Creation of APP
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Tool")

st.title("🏥 Clinical Heart Disease Predictor")
st.write("Enter the patient's clinical metrics to evaluate the probability of heart disease.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 45)
    chol = st.number_input("Cholesterol Level (mg/dl)", 100, 500, 200)
    fbs = st.selectbox("fasting blood sugar over 120 mg", ["yes","no"])
    thal = st.selectbox(
    "Thalassemia Results",
    ["Normal", "Fixed Defects", "Reversible Defect"],
    help="Blood disorder status based on the thallium scintigraphy stress test."
)

with col2:
    cp = st.slider("Chest Pain (1-4)",1,4,2)
    bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    restecg = st.selectbox("Electrical Acivity of Heart",["Normal","Abnormal"," probable or definite left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart-Rate Achieved",26,600,60)

with col3:
    ca = st.slider("flourosopy colored major vessels (0-3)",0,3,1)
    exang = st.selectbox("Exersize induced Angina", ["yes","no"])
    oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    0, 30, 3,
    help="ST depression induced by exercise relative to rest. High values are a strong indicator of coronary issues."
)
    slope = st.selectbox("ST segment", 
                         ["Upsloping","Flat","Downsloping"], 
                         help = "the interval between ventricular depolarization (contraction) and repolarization (relaxation)")

# Tranform responses
if sex == "Male":
    sex_val = 1
else:  
    sex_val = 0

if restecg == "Normal":
    restecg_val = 0
elif restecg == "Abnormal":
    restecg_val = 1
else:
    restecg_val = 2

if exang == "yes":
    exang_val = 1
else:
    exang_val = 0

if slope == "Upsloping":
    slope_val = 1
elif slope == "Flat":
    slope_val = 2
else:
    slope_val = 3

if thal == "Normal":
    thal_val = 3.0
elif thal == "Fixed Defects":
    thal_val = 6.0
else:
    thal_val = 7.0
    
if fbs == "yes":
    fbs_val = 1
else:
    fbs_val = 0

patient_data = {
    'age': age,
    'sex': sex_val,     
    'cp': cp,       
    'trestbps': bp,
    'chol': chol,
    'fbs': fbs_val,
    'restecg': restecg_val,
    'thalach': thalach,
    'exang': exang_val,
    'oldpeak': oldpeak,
    'slope': slope_val,
    'ca': ca,       
    'thal': 7
}

result = heart_disease_risk_calculator(patient_data)

if st.button("Analyze Risk"):
    result = heart_disease_risk_calculator(patient_data)
    
    st.divider()
    st.subheader("📊 Clinical Dashboard: Patient vs. Dataset Average")

    # 1. Top Level Metrics
    m1, m2, m3 = st.columns(3)
    
    # Calculate differences to show "Delta"
    #averages
    bp_delta = bp - avg_restingbp
    chol_delta = chol - avg_chols
    hr_delta = thalach - avg_hr

    m1.metric("Resting BP", f"{bp} mmHg", delta=f"{bp_delta} vs avg", delta_color="inverse")
    m2.metric("Cholesterol", f"{chol} mg/dl", delta=f"{chol_delta} vs avg", delta_color="inverse")
    m3.metric("Max Heart Rate", f"{thalach} bpm", delta=f"{hr_delta} vs avg")

    st.write("---")

    # 2. Comparison Bar Chart
    st.write("### Feature Comparison")
    
    # Prepare data for the chart
    chart_data = pd.DataFrame({
        "Metric": ["Age", "Resting BP", "Cholesterol", "Max HR"],
        "Patient": [age, bp, chol, thalach],
        "Dataset Average": [avg_age, avg_restingbp, avg_chols, avg_hr]
    }).set_index("Metric")

    # Display the chart
    st.bar_chart(chart_data)

    # 3. Final Interpretative Summary
    if result > 0.5:
        st.error(f"**High Risk Identified ({result:.1%}):**.")
    else:
        st.success(f"**Low Risk Identified ({result:.1%}):**.")
