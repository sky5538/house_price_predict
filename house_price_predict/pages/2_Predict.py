import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
import os

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏡 Malaysia House Price Prediction")

# ------------------------------
# GitHub 上模型的 raw 链接
# ------------------------------
MODEL_BASE_URL = "https://github.com/sky5538/house_price_predict/raw/main/house_price_predict/models/"

model_files = {
    "Linear Regression": "lin_pipeline.pkl",
    "Random Forest": "rf_model.pkl",
    "Gradient Boosting": "gb_pipeline.pkl"
}

# ------------------------------
# 加载模型函数
# ------------------------------
@st.cache_data(show_spinner=True)
def load_model_from_github(file_name):
    url = MODEL_BASE_URL + file_name
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# ------------------------------
# 加载所有模型
# ------------------------------
models = {}
for name, file in model_files.items():
    try:
        models[name] = load_model_from_github(file)
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")

selected_model_name = st.selectbox("Choose Model:", list(models.keys()))
chosen_model = models[selected_model_name]

# ------------------------------
# 用户输入
# ------------------------------
township = st.text_input("Enter Township:")
area = st.text_input("Enter Area:")
median_psf = st.number_input("Enter Price per square feet (RM):", min_value=0.0, step=10.0)

states = ["Johor","Kedah","Kelantan","Malacca","Negeri Sembilan","Pahang","Penang","Perak","Perlis",
          "Sabah","Sarawak","Selangor","Terengganu","Kuala Lumpur","Labuan","Putrajaya"]
state = st.selectbox("Select State:", states)

tenure = st.radio("Select Tenure:", ["Freehold","Leasehold"])

house_types = ["Terrace House","Cluster House","Semi D","Bungalow",
               "Service Residence","Flat","Town House","Apartment","Condominium"]
house_type = st.selectbox("Select House Type:", house_types)

# ------------------------------
# 预测按钮
# ------------------------------
if st.button("Predict Price"):
    try:
        # 构建输入 DataFrame
        new_data = pd.DataFrame([{
            "Township": township,
            "Area": area,
            "State": state,
            "Tenure": tenure,
            "Type": house_type,
            "Median_PSF": median_psf
        }])

        # ------------------------------
        # 预测
        # ------------------------------
        predicted_price = chosen_model.predict(new_data)

        # ------------------------------
        # 显示结果和图片
        # ------------------------------
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("💰 Prediction Result")
            st.success(f"Predicted Price using {selected_model_name}: RM {predicted_price[0]:,.2f}")
        with col2:
            # 本地图片路径
            local_image_path = os.path.join("house_images", f"{house_type}.jpg")
            if os.path.exists(local_image_path):
                st.image(local_image_path, caption=house_type, use_container_width=True)
            else:
                # GitHub 上 house_images 的 raw 链接
                IMAGE_BASE_URL = "https://github.com/sky5538/house_price_predict/raw/main/house_price_predict/house_images/"
                img_url = IMAGE_BASE_URL + f"{house_type}.jpg"
                st.image(img_url, caption=house_type, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
