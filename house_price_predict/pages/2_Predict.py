import streamlit as st
import pandas as pd
import os

st.title("🏡 Predict Malaysia House Price")

# -------------------------------
# 检查模型是否训练完成
# -------------------------------
if "trained_models" not in st.session_state or "lin_columns" not in st.session_state:
    st.error("⚠️ Please train models first in 'Train & Evaluate' page.")
    st.stop()

models = st.session_state["trained_models"]
lin_columns = st.session_state["lin_columns"]

selected_model_name = st.selectbox("Choose Model:", list(models.keys()))
chosen_model = models[selected_model_name]

# -------------------------------
# 用户输入
# -------------------------------
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

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Price"):
    try:
        new_data = pd.DataFrame([{
            "Township": township,
            "Area": area,
            "State": state,
            "Median_PSF": median_psf,
            "Tenure": tenure,
            "Type": house_type
        }])

        # -------------------------------
        # Linear Regression 自动补 OHE 列
        # -------------------------------
        if selected_model_name == "Linear Regression":
            for col in lin_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            # 设置 tenure/type 对应列为 1
            new_data[f"ohe__Tenure_{tenure}"] = 1
            new_data[f"ohe__Type_{house_type}"] = 1

            # 保证列顺序
            new_data = new_data[lin_columns]

        # -------------------------------
        # Random Forest / Gradient Boosting 自动补 Transactions 列
        # -------------------------------
        elif selected_model_name in ["Random Forest", "Gradient Boosting"]:
            # 预测直接用 pipeline，pipeline 会处理 OHE
            pass

        # -------------------------------
        # 预测
        # -------------------------------
        predicted_price = chosen_model.predict(new_data)

        # -------------------------------
        # 显示结果和图片
        # -------------------------------
        col1, col2 = st.columns([1,1])
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
