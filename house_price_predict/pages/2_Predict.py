import os
import streamlit as st
import pandas as pd
import joblib

st.title("🏡 Predict Malaysia House Price")

# ------------------------------
# 加载模型
# ------------------------------
try:
    lin_pipeline = joblib.load("models/lin_pipeline.pkl")  # Linear Regression + OHE
    rf_pipeline = joblib.load("models/rf_model.pkl")
    gb_pipeline = joblib.load("models/gb_pipeline.pkl")
    lin_columns = joblib.load("models/lin_columns.pkl")   # Linear Regression 的列顺序
except:
    st.error("⚠️ Please train models first in 'Train & Evaluate' page. Make sure 'models/' folder exists.")
    st.stop()

models = {
 "Random Forest": rf_pipeline,
    "Linear Regression": lin_pipeline,
   
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

        # ------------------------------
        # Linear Regression (OHE) 自动补列
        # ------------------------------
        if selected_model_name == "Linear Regression":
            # 补齐训练时的所有列
            for col in lin_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            # 将 tenure/type 对应列设为 1
            new_data[tenure] = 1
            new_data[house_type] = 1

            # 保证列顺序
            new_data = new_data[lin_columns]

        # ------------------------------
        # Random Forest / Gradient Boosting 自动补列
        # ------------------------------
        elif selected_model_name in ["Random Forest", "Gradient Boosting"]:
            # 构建 OHE 列
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
