
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model = joblib.load('best_heart_disease_model.pkl')

# Pre-trained LabelEncoders and Scalers
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Define the list of features expected by the model
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Function to preprocess input data
def preprocess_input(data):
    # Apply Label Encoding
    for col, le in label_encoders.items():
        data[col] = le.transform([data[col]])[0]
    
    # Convert to DataFrame for scaler
    data_df = pd.DataFrame(data, index=[0])
    
    # Apply scaling
    data_scaled = scaler.transform(data_df)
    
    return data_scaled

# Streamlit App
st.title("Heart Disease Prediction")

st.write("""
    This app predicts the likelihood of a patient having heart disease based on their medical details.
    Fill in the details below to get a prediction.
""")

# Input fields for user to enter patient data
input_data = {}
input_data['age'] = st.number_input('Age', min_value=1, max_value=120, value=30)
input_data['sex'] = st.selectbox('Sex', ['male', 'female'])
input_data['cp'] = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
input_data['trestbps'] = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
input_data['chol'] = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
input_data['fbs'] = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
input_data['restecg'] = st.selectbox('Resting Electrocardiographic Results', ['normal', 'having ST-T wave abnormality', 'showing probable or definite left ventricular hypertrophy'])
input_data['thalach'] = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
input_data['exang'] = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
input_data['oldpeak'] = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=6.0, step=0.1, value=1.0)
input_data['slope'] = st.selectbox('Slope of the peak exercise ST segment', ['upsloping', 'flat', 'downsloping'])
input_data['ca'] = st.number_input('Number of major vessels (0-3) colored by fluoroscopy', min_value=0, max_value=3, value=0)
input_data['thal'] = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversible defect'])

# Button to make prediction
if st.button('Predict'):
    # Preprocess input data
    input_data_preprocessed = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_preprocessed)
    
    # Display result
    if prediction[0] == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")

    # Display the prediction probability
    prediction_proba = model.predict_proba(input_data_preprocessed)
    st.write(f"Prediction Probability: {prediction_proba[0][1]*100:.2f}%")
