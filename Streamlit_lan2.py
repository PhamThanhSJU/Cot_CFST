import math
import pandas as pd
import pandas
import numpy as np
import streamlit as st
# Information
st.sidebar.markdown("## Thông tin tác giả")
st.sidebar.markdown("Tác giả: Nhóm NCKH khoa Công trình")
st.sidebar.caption("Khoa Công trình, ĐH Thủy Lợi")
st.sidebar.caption("Email: thanhpv@tlu.edu.com")

#Tên bài báo
st.title ("Dự đoán và phân tích ảnh hưởng của các tham số đầu vào đến khả năng chịu nén đúng tâm của cột thép nhồi bê tông tiết diện chữ nhật bằng mô hình học máy lai") 

# Chèn sơ đồ nghiên cứu
# st.header("1. Layout of this study")
# check1=st.checkbox('1.1 Display layout of this investigation')
# if check1:
#    st.image("Fig. 1.jpg", caption="Layout of this study")
# # Hiển thị dữ liệu
# st.header("2. Dataset")
# check2=st.checkbox('2.1 Display dataset')
# if check2:
#    Database="Dataset.csv"
#    df = pandas.read_csv(Database)
#    df.head()
#    st.write(df)
# st.header("3. Modeling approach")
# check3=st.checkbox('3.1 Display structure of Random Forest model')
# if check3:
#    st.image("Fig2.jpg", caption="Overview on structure of Random Forest model") 
	
#Make a prediction
st.header("Dự đoán khả năng chịu nén đúng tâm của cột CFST chữ nhật bằng PSO-XGB")
st.subheader("Các biến đầu vào:")
col1, col2 =st.columns(2)

with col1:
   X1=st.slider("Chiều rộng b(mm)", 62, 319)
   X2 = st.slider("Chiều cao h(mm)", 44, 319)
   X3= st.slider("Độ dày ống thép t(mm)", 2, 12)

	
with col2:
   X4 = st.slider("Chiều dài cột L(mm)", 60, 2940)	
   X5 = st.slider("Giới hạn chảy của thép fy(MPa)", 115, 835)
   X6 = st.slider("Cường độ nén bê tông fc(MPa)", 10, 148)

# with col3:	
   
#    X10 = st.slider("B /1000 (wt. %)", 0.00, 5.70)	
#    X11 = st.slider("P /100 (wt. %)", 0.10, 4.00)
#    X12 = st.slider("S /100 (wt. %)", 0.40, 1.90)
#    X13 = st.slider("Al (wt. %)", 0.00, 0.10)	
# with col4:	
#    X15 = st.slider("Nb /100 (wt. %)", 0.00, 1.00)	
#    X17 = st.slider("Pb /100 (wt. %)", 0.00, 2.00)
#    X19 = st.slider("Pre-irr. flue. x10^22(n/cm2)", 0.00, 3.86)		
# with col5:	
#    X20 = st.slider("Irr. flue. x10^22(n/cm2)", 1.53, 14.00)
#    X21 = st.slider("Temperature (C)", 376.00, 780.10)  
# with col6:	
#    X22 = st.slider("Stress (MPa)", 0.0, 403.0)
#    X23 = st.slider("Disl. Dens. x10^14(1/m2)", 1.50, 30.00)


from sklearn.model_selection import train_test_split, KFold
#from sklearn.ensemble import GradientBoostingRegressor
#import catboost as cb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
#Model

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# df = pd.read_csv('Swelling_data_316SS_Thanh_WERCS2.csv')
# # X_ori = df[['D','W','den','fc','E','L','C','P']].values
# X_ori = df[['Fe','Cr','Mn','Si','Co','Mo','C','N','B','P','S','Al','Nb','Pb','Pre-irr','Irr','Temp','Stre','Dis']].values
# y = df['Swell'].values

df = pd.read_csv("Data_chuan3.csv")
X_ori = df[["b", "h", "t", "L", "fy", "fc"]].values
y = df["N"].values

X = scaler.fit_transform(X_ori)	
from numpy import asarray

Inputdata = [X1, X2, X3, X4, X5, X6]

X_predict = asarray([Inputdata])
predictions = []
  
for i in range(1,5,1):
    count = i
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=(0+i))

    # n_estimators = 502
    # learning_rate = 0.223
    # depth = 4
    # l2_leaf_reg = 7
    
    #     model = XGBRegressor(
    #         n_estimators=int(p1),
    #         max_depth=int(p2),
    #         learning_rate=p3,
    #         objective="reg:squarederror",
    #         verbosity=0
    #     )


    
    # cat_clf_n = CatBoostRegressor(n_estimators=n_estimators,learning_rate = learning_rate, depth=depth,l2_leaf_reg=l2_leaf_reg, random_state=42)
    # cat_clf_n.fit(X_train, y_train)
    
    cat_clf_n = XGBRegressor(
        n_estimators=218,
        max_depth=4,
        learning_rate=0.2,
        objective="reg:squarederror",
        verbosity=0
    )
    # CatBoostRegressor(n_estimators=n_estimators,learning_rate = learning_rate, depth=depth,l2_leaf_reg=l2_leaf_reg, random_state=42)
    cat_clf_n.fit(X_train, y_train)


    # X_predict1= scaler.fit_transform(X_predict)
    X_predict1=scaler.transform(X_predict)

    result_xgb = cat_clf_n.predict(X_predict1)

    predictions.append(result_xgb)
	
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)

puxgb = np.around(mean_pred, 3)
puxgb_err = np.around(std_pred, 3)

#fc_pred2 = cat_clf_n.predict(Newdata)

st.subheader("Dự đoán khả năng chịu nén đúng tâm")
if st.button("Dự đoán"):
    import streamlit as st
    import time
    progress_text = "Đang dự đoán. Vui lòng chờ trong giây lát."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(f"Giá trị dư đoán trung bình trong 5 lần từ PSO-XGB được là: {(puxgb)}")
    st.success(f"Giá trị độ lệch chuẩn trong 5 lần dự đoán từ PSO-XGB là: {(puxgb_err)}")
