import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import joblib
import os

st.title("🏡 Malaysia House Price Prediction")

# ------------------------------
# GitHub raw links
# ------------------------------
GITHUB_BASE = "https://github.com/sky5538/house_price_predict/raw/main/house_price_predict/models/"

MODEL_FILES = {
    "Linear Regression": "lin_pipeline.pkl",
    "Random Forest": "rf_model.pkl",
    "Gradient Boosting": "gb_pipeline.pkl",
    "Linear Columns": "lin_columns.pkl"
}

# ------------------------------
# Function to load model from GitHub
# ------------------------------
@st.cache_data(show_spinner=True)
def load_model(file_name):
    url = GITHUB_BASE + file_name
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# ------------------------------
# Load models
# ------------------------------
try:
    lin_pipeline = load_model(MODEL_FILES["Linear Regression"])
    rf_pipeline = load_model(MODEL_FILES["Random Forest"])
    gb_pipeline = load_model(MODEL_FILES["Gradient Boosting"])
    lin_columns = load_model(MODEL_FILES["Linear Columns"])
except Exception as e:
    st.error(f"⚠️ Failed to load models from GitHub: {e}")
    st.stop()

models = {
    "Random Forest": rf_pipeline,
    "Linear Regression": lin_pipeline,
    "Gradient Boosting": gb_pipeline
}

selected_model_name = st.selectbox("Choose Model:", list(models.keys()))
chosen_model = models[selected_model_name]

# ------------------------------
# User inputs
# ------------------------------
township = st.text_input("Enter Township:")
area = st.text_input("Enter Area:")
median_psf = st.number_input("Enter Price per square feet (RM):", min_value=0.0, step=10.0)

states = ["Johor","Kedah","Kelantan","Malacca","Negeri Sembilan","Pahang","Penang",
          "Perak","Perlis","Sabah","Sarawak","Selangor","Terengganu",
          "Kuala Lumpur","Labuan","Putrajaya"]
state = st.selectbox("Select State:", states)

tenure = st.radio("Select Tenure:", ["Freehold","Leasehold"])

house_types = ["Terrace House","Cluster House","Semi D","Bungalow",
               "Service Residence","Flat","Town House","Apartment","Condominium"]
house_type = st.selectbox("Select House Type:", house_types)

# ------------------------------
# Predict button
# ------------------------------
if st.button("Predict Price"):
    try:
        # ------------------------------
        # Standardize user input capitalization
        # ------------------------------
        township = township.strip().title()
        area = area.strip().title()
        state = state.strip().title()
        tenure = tenure.strip().title()
        house_type = house_type.strip().title()

        # Construct base input
        new_data = pd.DataFrame([{
            "Township": township,
            "Area": area,
            "State": state,
            "Median_PSF": median_psf
        }])

        # ------------------------------
        # Linear Regression auto-fill columns
        # ------------------------------
        if selected_model_name == "Linear Regression":
            for col in lin_columns:
                if col not in new_data.columns:
                    new_data[col] = 0
            new_data[tenure] = 1
            new_data[house_type] = 1
            new_data = new_data[lin_columns]

        # ------------------------------
        # Random Forest / Gradient Boosting auto-fill columns
        # ------------------------------
        elif selected_model_name in ["Random Forest", "Gradient Boosting"]:
            tenure_cols = ["Freehold", "Leasehold"]
            type_cols = ["Terrace House","Cluster House","Semi D","Bungalow",
                         "Service Residence","Flat","Town House","Apartment","Condominium"]
            ohe_dict = {col: 0 for col in tenure_cols + type_cols}
            ohe_dict[tenure] = 1
            ohe_dict[house_type] = 1
            ohe_dict["Transactions"] = 0  # If training data has Transactions column
            new_data = pd.DataFrame([{
                "Township": township,
                "Area": area,
                "State": state,
                "Median_PSF": median_psf,
                **ohe_dict
            }])

        # ------------------------------
        # Predict
        # ------------------------------
        predicted_price = chosen_model.predict(new_data)

        # ------------------------------
        # Display results and image
        # ------------------------------
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("💰 Prediction Result")
            st.success(f"Predicted Price using {selected_model_name}: RM {predicted_price[0]:,.2f}")

        with col2:
            IMAGE_BASE = "https://github.com/sky5538/house_price_predict/raw/main/house_price_predict/house_images/"
            image_url = IMAGE_BASE + f"{house_type}.jpg"
            try:
                st.image(image_url, caption=house_type, use_container_width=True)
            except:
                st.info("ℹ️ No image available for this house type.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
