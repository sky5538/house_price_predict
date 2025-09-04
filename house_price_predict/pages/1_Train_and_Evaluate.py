import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

st.title("🏡 Train & Evaluate Models")

# 上传训练与测试数据
train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
test_file = st.file_uploader("Upload Testing Data (CSV)", type=["csv"])

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    target_col = "Price"

    # 自动识别分类列和数值列
    categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in train_df.columns if c not in categorical_cols + [target_col]]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # -------------------------------
    # ColumnTransformer + Pipeline
    # -------------------------------
    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    # Linear Regression Pipeline
    lin_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lin_pipeline.fit(X_train, y_train)
    y_pred_lin = lin_pipeline.predict(X_test)

    # Random Forest Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    # Gradient Boosting Pipeline
    gb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ])
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)

    # -------------------------------
    # 评估函数（RMSE、MAE 归一化，NMAE 显示 %）
    # -------------------------------
    def evaluate(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mean_y = y_true.mean()
        nrmse = rmse / mean_y      # 归一化 RMSE
        nmae = mae / mean_y * 100  # NMAE %
        return {
            "Model": name,
            "R²": round(r2, 4),
            "RMSE": round(nrmse, 4),
            "MAE (%)": round(nmae, 2)
        }

    # -------------------------------
    # 汇总结果
    # -------------------------------
    results = [
        evaluate(y_test, y_pred_lin, "Linear Regression "),
        evaluate(y_test, y_pred_rf, "Random Forest "),
        evaluate(y_test, y_pred_gb, "Gradient Boosting ")
    ]

    st.subheader("📑 Model Performance Comparison")
    st.dataframe(pd.DataFrame(results))

    # -------------------------------
    # 保存模型
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(lin_pipeline, "models/lin_pipeline.pkl")
    joblib.dump(rf_pipeline, "models/rf_pipeline.pkl")
    joblib.dump(gb_pipeline, "models/gb_pipeline.pkl")

    st.success("✅ Models trained.")
