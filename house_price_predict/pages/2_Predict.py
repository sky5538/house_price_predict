import os
import streamlit as st
import pandas as pd

st.title("🏡 Predict Malaysia House Price")

# ------------------------------
# 从 session_state 获取模型
# ------------------------------
if "trained_models" in st.session_state:
    models = st.session_state["trained_models"]
else:
    st.error("⚠️ Please train models first in 'Train & Evaluate' page.")
    st.stop()

# 模型名称选择
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
        # Linear Regression (OHE)
        # ------------------------------
        if selected_model_name == "Linear Regression":
            # session_state 里存的列顺序
            lin_columns = st.session_state.get("lin_columns", list(new_data.columns))
            # 补齐缺失列
            for col in lin_columns:
                if col not in new_data.columns:
                    new_data[col] = 0
            new_data[tenure] = 1
            new_data[house_type] = 1
            # 保证列顺序
            new_data = new_data[lin_columns]

        # ------------------------------
        # Random Forest / Gradient Boosting
        # ------------------------------
        else:
            tenure_cols = ["Freehold", "Leasehold"]
            type_cols = ["Terrace House","Cluster House","Semi D","Bungalow",
                         "Service Residence","Flat","Town House","Apartment","Condominium"]
            ohe_dict = {col: 0 for col in tenure_cols + type_cols}
            ohe_dict[tenure] = 1
            ohe_dict[house_type] = 1
            ohe_dict["Transactions"] = 0  # 自动补 Transactions 列

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
