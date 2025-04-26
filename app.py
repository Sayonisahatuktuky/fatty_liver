import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load('model.pkl')

# Title for Streamlit app
st.set_page_config(page_title="Fatty Liver Disease Prediction", page_icon="🩺")

# Web App Main Page
def main():
    st.title("🩺 Fatty Liver Disease Prediction")

    st.write("### Enter patient health details below:")

    # Input fields
    age = st.number_input("Age", 1, 100, 30)
    alt = st.number_input("ALT", 0, 200, 20)
    ast = st.number_input("AST", 0, 200, 22)
    chol = st.number_input("Cholesterol", 100, 400, 180)
    tg = st.number_input("Triglycerides", 50, 500, 150)
    hdl = st.number_input("HDL", 20, 100, 45)
    ldl = st.number_input("LDL", 30, 300, 100)
    ggt = st.number_input("GGT", 0, 300, 30)

    if st.button("Predict"):
        input_data = np.array([[age, alt, ast, chol, tg, hdl, ldl, ggt]])
        
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Fatty Liver Disease!")
        else:
            st.success("✅ Low Risk of Fatty Liver Disease.")

# Run app
if __name__ == '__main__':
    main()
