import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="CFST Predictor")

# ===== SIDEBAR =====
st.sidebar.markdown("## Thông tin tác giả")
st.sidebar.markdown("Tác giả: Nhóm NCKH khoa Công trình")
st.sidebar.caption("Khoa Công trình, ĐH Thủy Lợi")
st.sidebar.caption("Email: thanhpv@tlu.edu.vn")

st.title("Dự đoán CFST (Bootstrap 90%)")

# ===== DATA =====
df = pd.read_csv("Data_chuan3.csv")
X_full = df[["b", "h", "t", "L", "fy", "fc"]]
y_full = df["N"]

# ===== INPUT =====
labels = [
    ("b (mm)", 62, 319),
    ("h (mm)", 44, 319),
    ("t (mm)", 2, 12),
    ("L (mm)", 60, 2940),
    ("fy (MPa)", 115, 835),
    ("fc (MPa)", 10, 148),
]

cols = st.columns(2)
x_input = [(cols[i % 2]).slider(n, a, b) for i, (n, a, b) in enumerate(labels)]

# ===== PREDICT =====
if st.button("Dự đoán"):

    preds = []

    for _ in range(5):  # 5 lần bootstrap
        # lấy 90% dữ liệu train ngẫu nhiên
        idx = np.random.choice(len(df), int(0.9 * len(df)), replace=False)

        X_train = X_full.iloc[idx]
        y_train = y_full.iloc[idx]

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = XGBRegressor(
            n_estimators=218,
            max_depth=4,
            learning_rate=0.2,
            objective="reg:squarederror",
            verbosity=0
        )
        model.fit(X_train_s, y_train)

        X_test = scaler.transform([x_input])
        preds.append(model.predict(X_test)[0])

    preds = np.array(preds)

    st.success(f"Giá trị trung bình: {preds.mean():.3f} kN")
    st.info(f"Độ lệch chuẩn: {preds.std():.3f} kN")
