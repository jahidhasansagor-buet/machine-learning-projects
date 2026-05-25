import streamlit as st
import pandas as pd
import joblib

# Load the pipeline once at startup
pipeline = joblib.load('used-car-price-prediction/models/best_model.pkl')

st.title("Used Car Price Predictor")
st.write("Fill in the details below and click the button to get an estimated selling price.")

st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, max_value=100.0, step=0.1, value=5.0)
    kms_driven = st.number_input("Kilometres Driven", min_value=0, max_value=200000, step=500, value=30000)
    car_age = st.slider("Car Age (years)", min_value=1, max_value=20, value=5)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.markdown("---")

# Predict button
if st.button("Predict Selling Price"):
    input_df = pd.DataFrame([{
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Car_Age': car_age,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner,
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction = max(0, prediction)  # just to avoid any negative output

    st.success(f"Estimated Selling Price: {prediction:.2f} Lakhs")
    st.info(f"Note: This is an estimate based on {301} training samples from the CarDekho dataset.")
