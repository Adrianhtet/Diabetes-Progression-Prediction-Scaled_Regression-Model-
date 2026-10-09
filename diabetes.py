import streamlit as st
import pickle

st.title("Diabetes Progression Prediction (Scaled_Regression Model)")

st.sidebar.header("Input Patient Features")

age = st.sidebar.number_input("Age", value=0.0, format="%.4f")
sex = st.sidebar.number_input("Sex", value=0.0, format="%.4f")
bmi = st.sidebar.number_input("BMI", value=0.0, format="%.4f")
bp = st.sidebar.number_input("Blood Pressure", value=0.0, format="%.4f")

s1 = st.sidebar.number_input("S1", value=0.0, format="%.4f")
s2 = st.sidebar.number_input("S2", value=0.0, format="%.4f")
s3 = st.sidebar.number_input("S3", value=0.0, format="%.4f")
s4 = st.sidebar.number_input("S4", value=0.0, format="%.4f")
s5 = st.sidebar.number_input("S5", value=0.0, format="%.4f")
s6 = st.sidebar.number_input("S6", value=0.0, format="%.4f")

if st.button("Predict"):
    data = [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]]

    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)

    result = model.predict(data)[0]

    st.write("Predicted Diabetes Progression Score:")
    st.write(round(result, 2))

