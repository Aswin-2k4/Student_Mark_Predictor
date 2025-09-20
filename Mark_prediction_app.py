
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np


st.title("Student performace Predictor")
st.write("Enter the following")

study=st.number_input("Enter study hours :",max_value=14)
social=st.number_input("Enter social media hours :",max_value=12)
attend=st.number_input("Enter attendence percent :",max_value=100)
sleep=st.number_input("Enter sleep hours :",max_value=10)
exer=st.number_input("Enter exercise frequency(ranging from 0-6) :",step=1,min_value=0,max_value=6)
mental=st.number_input("Enter mental health rating outoff 10 :",step=1,min_value=0,max_value=10)


input_lt=[[study,social,attend,sleep,exer,mental]]

scaler=joblib.load(r'C:\Users\HP\OneDrive\Documents\PJT2.0\scaler.pkl')
input_lt=scaler.transform(input_lt)


model=joblib.load(r'C:\Users\HP\OneDrive\Documents\PJT2.0\score_model.pkl')
if st.button("Predict"):
   score=model.predict(input_lt)
   score=np.clip(score,0,100)
   score=np.round(score,2)
   score=int(score)
   st.write(f"Predicted Exam Score : {score}/100")
