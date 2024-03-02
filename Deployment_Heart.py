import streamlit as st

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('Heart_model.sav','rb'))

def check(input_data):

    array_input = np.array(input_data)

    reshaped_input = array_input.reshape(1,-1)

    prediction = loaded_model.predict(reshaped_input)

    return (prediction[0])

def main():
    st.title("Heart Attack Prediction")

    age = st.text_input("Age")

    gender = st.text_input("Gender")

    impluse = st.text_input("Impluse")

    pressurehight = st.text_input("Pressure High")

    pressurelow = st.text_input("Pressure Low")

    glucose = st.text_input("Glucose")

    kcm = st.text_input("KCM")

    troponin = st.text_input("Troponin")

    pred = ""
    if st.button("Click Here for Result Prediction"):
        pred = check([age,gender,impluse,pressurehight,pressurelow,glucose,kcm,troponin])

    st.success(f"Your Heart Attack Chances is {pred} ")

if __name__=='__main__':
    main()
    