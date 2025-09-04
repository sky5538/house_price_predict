import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.title("📊 Train & Evaluate Models")

# Upload train + test dataset
train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
test_file = st.file_uploader("Upload Testing Dataset (CSV)", type=["csv"])

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # 自动找出可选的 attributes（去掉目标变量和不需要的）
    default_drop = ["Price"]  # Price 是目标变量
    all_features = [col for col in train_df.columns if col not in default_drop]

    # 让用户选择 attributes
    selected_features = st.multiselect(
        "Select features to use for training:",
        options=all_features,
        default=[col for col in all_features if col not in ["Township", "Transactions"]]
    )

    if selected_features:
        X_train = train_df[selected_features]
        y_train = train_df["Price"]
        X_test = test_df[selected_features]
        y_test = test_df["Price"]

        # 检查是否包含分类特征
        categorical_features = [col for col in selected_features if train_df[col].dtype == "object"]

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough"
        )

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }

        results = []
        trained_models = {}

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            results.append({
                "Model": name,
                "R²": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred)
            })

            trained_models[name] = pipeline

        st.subheader("📑 Model Performance Comparison")
        st.dataframe(pd.DataFrame(results))

        # Save models into session_state
        st.session_state["trained_models"] = trained_models
        st.success("✅ Models trained and saved. Go to Predict page to use them.")
    else:
        st.warning("⚠️ Please select at least one feature to train the model.")
