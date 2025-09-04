import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os

st.title("🏡 Predict Malaysia House Price")

# ------------------------------
# 从 GitHub 下载模型
# ------------------------------
def load_model_from_github(url, model_name):
    if model_name not in st.session_state:
        r = requests.get(url)
        r.raise_for_status()
        st.session_state[model_name] = joblib.load(io.BytesIO(r.content))
    return st.session_state[model_name]

# GitHub raw 链接
url_lin = "https://raw.githubusercontent.com/sky5538/house_price_predict/main/house_price_predict/models/lin_pipeline.pkl"
url_rf = "https://raw.githubusercontent.com/sky5538/house_price_predict/main/house_price_predict/models/rf_pipeline.pkl"
url_gb = "https://raw.githubusercontent.com/sky5538/house_price_predict/main/house_price_predict/models/gb_pipeline.pkl"

lin_pipeline = load_model_from_github(url_lin, "lin_pipeline")
rf_pipeline = load_model_from_github(url_rf, "rf_pipeline")
gb_pipeline = load_model_from_github(url_gb, "gb_pipeline")

# 获取 Linear Regression 的列名
if "lin_columns" not in st.session_state:
    # 获取训练时 OneHotEncoder 的特征名 + 数值列
    lin_columns = list(lin_pipeline.named_steps["preprocessor"].get_feature_names_out())
    num_cols = lin_pipeline.named_steps["preprocessor"].remainder
    if num_cols == "passthrough":
        # 假设训练时数值列为 Median_PSF
        lin_columns += ["Median_PSF"]
    st.session_state["lin_columns"] = lin_columns
lin_columns = st.session_state["lin_columns"]

# ------------------------------
# 模型选择
# ------------------------------
models = {
    "Linear Regression": lin_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline
}

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
        # 构建基础输入
        new_data = pd.DataFrame([{
            "Township": township,
            "Area": area,
            "State": state,
            "Median_PSF": median_psf
        }])

        if selected_model_name == "Linear Regression":
            # 自动补齐 Linear Regression OHE 列
            for col in lin_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            # 将 tenure/type 对应列设为 1
            new_data[tenure] = 1
            new_data[house_type] = 1

            # 保证列顺序
            new_data = new_data[lin_columns]

        else:
            # Random Forest / Gradient Boosting 补齐 OHE 列
            tenure_cols = ["Freehold", "Leasehold"]
            type_cols = ["Terrace House","Cluster House","Semi D","Bungalow",
                         "Service Residence","Flat","Town House","Apartment","Condominium"]
            ohe_dict = {col: 0 for col in tenure_cols + type_cols}
            ohe_dict[tenure] = 1
            ohe_dict[house_type] = 1

            # 补 Transactions 列为 0
            ohe_dict["Transactions"] = 0

            new_data = pd.DataFrame([{
                "Township": township,
                "Area": area,
                "State": state,
                "Median_PSF": median_psf,
                **ohe_dict
            }])

        # ------------------------------
        # 预测
        # ------------------------------
        predicted_price = chosen_model.predict(new_data)

        # ------------------------------
        # 显示结果和图片
        # ------------------------------
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("💰 Prediction Result")
            st.success(f"Predicted Price using {selected_model_name}: RM {predicted_price[0]:,.2f}")
        with col2:
            image_path = os.path.join("house_images", f"{house_type}.jpg")
            if os.path.exists(image_path):
                st.image(image_path, caption=house_type, use_container_width=True)
            else:
                st.info("ℹ️ No image available for this house type.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
