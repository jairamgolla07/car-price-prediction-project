import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import datetime

# Set page configuration with a wide layout
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    try:
        return joblib.load("models/cardata")
    except Exception:
        booster = xgb.XGBRegressor()
        booster.load_model("models/xgb_model.json")
        return booster

model = load_model()

# --- Header ---
st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Enter your car's details to get an estimated selling price using our XGBoost ML model.")
st.divider()

# --- Prediction Form ---
with st.form("prediction_form"):
    # --- Input Columns ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Vehicle Details")
        present_price = st.number_input("Current Showroom Price (Lakhs)", min_value=0.5, value=7.85, step=0.1, format="%.2f", help="The original price of the car.")
        purchase_year = st.slider("Year of Purchase", min_value=2000, max_value=datetime.datetime.now().year, value=2010, step=1)
        kms_driven = st.slider("Kilometers Driven", min_value=0, max_value=200000, value=50000, step=1000)

    with col2:
        st.subheader("Technical Specifications")
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        transmission = st.radio("Transmission Type", ["Manual", "Automatic"], horizontal=True)

    with col3:
        st.subheader("Ownership & Seller")
        owner = st.slider("Number of Previous Owners", 0, 4, 1, help="How many times the car has been sold previously.")
        seller_type = st.radio("Seller Type", ["Individual", "Dealer"], horizontal=True)

    # Submit button for the form
    submitted = st.form_submit_button("Get Price Estimate", use_container_width=True)


# --- Prediction Logic & Output ---
if submitted:
    # Calculate age of the car
    age = datetime.datetime.now().year - purchase_year

    # Encoding categorical features
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    # Create a DataFrame for the model
    data_new = pd.DataFrame({
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_map[fuel]],
        "Seller_Type": [seller_map[seller_type]],
        "Transmission": [trans_map[transmission]],
        "Owner": [owner],
        "Age": [age]
    })

    # Predict the price
    with st.spinner('Calculating the best price for you...'):
        prediction = model.predict(data_new)
        predicted_price = prediction[0]

    st.divider()
    st.subheader("Prediction Result")

    # Display the result in columns
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(label="Estimated Selling Price", value=f"₹ {predicted_price:.2f} Lakhs")
        st.balloons()

    with col_res2:
        with st.expander("Show Prediction Inputs", expanded=True):
            st.write(f"**Showroom Price:** ₹ {present_price} Lakhs")
            st.write(f"**Year of Purchase:** {purchase_year} ({age} years old)")
            st.write(f"**Kms Driven:** {kms_driven:,} km")
            st.write(f"**Fuel / Transmission:** {fuel} / {transmission}")
            st.write(f"**Ownership:** {owner} previous owner(s)")
            st.write(f"**Seller:** {seller_type}")