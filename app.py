import streamlit as st
import numpy as np 
import joblib
import os
st.markdown(
    """
    <style>
    body {
        background-size: cover;
        background-color: white;
    }
    .main-title {
        font-size: 50px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
    }
    .sub-title {
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 class='main-title'>Welcome to My Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>CardioShield: Early Prediction for a Safer Tomorrow</p>", unsafe_allow_html=True)

try:
    rf = joblib.load(os.path.join('Random Forest.pkl'))
    svm = joblib.load(os.path.join('Support Vector Machine.pkl'))
except FileNotFoundError:
    rf, svm = None, None, None  # Set to None if not found

def predict_heart_disease(input_data, model):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

sex_options = [0,1]
sex_labels = ['Female','Male']
sex = st.selectbox( label = "Sex" , options=sex_options , format_func= lambda x : sex_labels[x] )
age = st.slider(label= 'Enter Your Age :', min_value= 1 , max_value=100,value=25)


col1, col2 = st.columns(2)
cp_options = [3,1,2,0]
ecg_options = [1,2,0]
ecg_labels = ['Normal', 'ST', 'LVH']
cp_labels = ['Typical Angina' , 'Atypical Angina' , 'Non-Anginal pain','Asymptomatic']
with col1:
   cp = st.selectbox(label = "Chest Pain", options=cp_options, format_func=lambda x: cp_labels[x]) 
   rbp = st.number_input("Resting Blood Pressure", min_value=80 , max_value=200 , value=120)

with col2 :
   restecg = st.selectbox("Resting Electrocardiographic Results (RestECG", options=ecg_options , format_func= lambda x: ecg_labels[x])
   chol = st.number_input('Cholesterol (mg/dl)', min_value=100 , max_value=600 , value=200)

slope_options = [2,0,1]
slope_labels = ['Up', 'Down','Flat']
maxhr = st.slider("Maximium Heart Rate Achived" , min_value=60, max_value=202 , value=150) 
slope = st.selectbox("Slope of Peak Exercise ST Segment (Slope)", options = slope_options , format_func= lambda x: slope_labels[x])
oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
model_dict = {
    'Random Forest': rf,
    'SVM': svm
}
model_choice = st.selectbox("Which model do you want to use?", options=list(model_dict.keys()))
selected_model = model_dict[model_choice]
col3 ,col4 = st.columns(2)
with col3:
   fbs= 1 if st.checkbox("Do You Have Fasting Blood Suger > 120 mg/dl")==True else 0
with col4:
   xang= 1 if st.checkbox("Do You Have Exercise Induced Angina")== True else 0

 
if st.button('Predict'):
   
       
    if selected_model is None:
        st.error(f"The selected model ({model_choice}) is not available. Please ensure the model is trained and saved.")
    else:
            input_data = [age, sex, cp, rbp, chol, fbs, restecg, maxhr, xang, oldpeak, slope]
            prediction = predict_heart_disease(input_data, selected_model)
            
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error(f"Heart Disease Detected!")
            else:
                st.success(f"No Heart Disease Detected! ")
    #    else:
    #         st.warning("No model available! Please train a model ")

st.write("Disclaimer: This app is for educational purposes only and should not be used as a substitute for professional medical advice.")
