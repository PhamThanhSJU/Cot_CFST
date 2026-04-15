import numpy as np
import pandas as pd
import streamlit as st
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## Thông tin tác giả")
st.sidebar.markdown("Tác giả: Nhóm NCKH khoa Công trình")
st.sidebar.caption("Khoa Công trình, ĐH Thủy Lợi")
st.sidebar.caption("Email: thanhpv@tlu.edu.com")

# =========================
# TITLE
# =========================
st.title("Dự đoán khả năng chịu nén đúng tâm của cột CFST tiết diện chữ nhật bằng PSO-XGB")

# =========================
# INPUT
# =========================
st.header("Nhập thông số đầu vào")

col1, col2 = st.columns(2)

with col1:
    X1 = st.slider("Chiều rộng b (mm)", 60, 319, 150)
    X2 = st.slider("Chiều cao h (mm)", 44, 319, 150)
    X3 = st.slider("Độ dày ống thép t (mm)", 2, 12, 5)

with col2:
    X4 = st.slider("Chiều dài cột L (mm)", 60, 2940, 800)
    X5 = st.slider("Giới hạn chảy thép fy (MPa)", 115, 835, 400)
    X6 = st.slider("Cường độ bê tông fc (MPa)", 10, 148, 60)

Inputdata = np.array([[X1, X2, X3, X4, X5, X6]])

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("Data_chuan3.csv")
except:
    st.error("❌ Không tìm thấy file Data_chuan3.csv")
    st.stop()

X_ori = df[["b", "h", "t", "L", "fy", "fc"]].values
y = df["N"].values

# =========================
# PREDICTION FUNCTION
# =========================
def predict():
    predictions = []

    for i in range(1, 6):  # 5 lần chạy
        X_train, X_test, y_train, y_test = train_test_split(
            X_ori, y, test_size=0.10, random_state=i
        )

        # Chuẩn hóa đúng (fit trên train)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_input_scaled = scaler.transform(Inputdata)

        # Model XGB
        model = XGBRegressor(
            n_estimators=218,
            max_depth=4,
            learning_rate=0.2,
            objective="reg:squarederror",
            verbosity=0
        )

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_input_scaled)

        predictions.append(pred[0])

    return np.mean(predictions), np.std(predictions)

# =========================
# BUTTON
# =========================
st.header("Kết quả dự đoán")

if st.button("Dự đoán"):
    progress_text = "Đang tính toán..."
    my_bar = st.progress(0, text=progress_text)

    for percent in range(100):
        time.sleep(0.01)
        my_bar.progress(percent + 1, text=progress_text)

    my_bar.empty()

    mean_pred, std_pred = predict()

    st.success(f"✅ Giá trị dự đoán trung bình: {mean_pred:.2f} kN")
    st.info(f"📊 Độ lệch chuẩn: {std_pred:.2f} kN")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Ứng dụng dự đoán sử dụng mô hình học máy lai PSO–XGB")