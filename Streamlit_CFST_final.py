import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="CFST Predictor")

# ===== SIDEBAR =====
st.sidebar.markdown("## Thông tin tác giả")
st.sidebar.info("Nhóm NCKH khoa Công trình\nĐH Thủy Lợi\nthanhpv@tlu.edu.vn")

st.title("Dự đoán sức chịu nén CFST chữ nhật bằng mô hình lai (PSO-XGB)")

# ===== MODEL =====
@st.cache_resource
def load_model():
    df = pd.read_csv("Data_chuan3.csv")
    X, y = df.iloc[:, :6], df["N"]

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=218,
        max_depth=4,
        learning_rate=0.2,
        objective="reg:squarederror",
        verbosity=0
    ).fit(Xs, y)

    return model, scaler

model, scaler = load_model()

# ===== INPUT =====
st.header("Nhập thông số")

cols = st.columns(2)
labels = [
    ("b (mm)", 62, 319),
    ("h (mm)", 44, 319),
    ("t (mm)", 2, 12),
    ("L (mm)", 60, 2940),
    ("fy (MPa)", 115, 835),
    ("fc (MPa)", 10, 148),
]

values = [cols[i % 2].slider(name, mn, mx) for i, (name, mn, mx) in enumerate(labels)]

# ===== PREDICT =====
if st.button("Dự đoán"):
    pred = model.predict(scaler.transform([values]))[0]
    st.success(f"Kết quả: {pred:.3f} kN")
