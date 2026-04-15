import math
import pandas as pd
import pandas
import numpy as np
import streamlit as st
# Information
st.sidebar.markdown("## Authors' information")
st.sidebar.markdown("Authors: Van-Thanh Pham and Jong-Sung Kim")
st.sidebar.caption("Department of Quantum and Nuclear Engineering, Sejong University, Korea")
st.sidebar.caption("Emails: phamthanhwru@gmail.com and kimjsbat@sejong.ac.kr")

#Tên bài báo
st.title ("Hybrid machine learning model using HHO algorithm for predicting the onset of void swelling in irradiated metals") 

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
st.header("Predicting the onset of void swelling in irradiated metals")
st.subheader("Input variables")
col1, col2, col3, col4, col5, col6 =st.columns(6)
with col1:
   X1=st.slider("Fe (wt. %)", 0.00, 97.00)
   X2 = st.slider("Cr (wt. %)", 0.00, 24.70)
   X3= st.slider("Mn (wt. %)", 0.00, 20.00)
   X4 = st.slider("Si (wt. %)", 0.00, 1.50)
   
	
with col2:	
   X5 = st.slider("Co /100 (wt. %)", 0.00, 4.00)
   X6= st.slider("Mo (wt. %)", 0.00, 2.95)
   X8 = st.slider("C (wt. %)", 0.00, 1.00)	
   X9 = st.slider("Ti (wt. %)", 0.0, 2.20)


with col3:		
   X11 = st.slider("B /1000 (wt. %)", 0.00, 4.00)
   X12 = st.slider("P /100 (wt. %)", 0.00, 15.50)
   X13= st.slider("S /100 (wt. %)", 0.00, 3.00)
   X14 = st.slider("Nb (wt. %)", 0.00, 0.92)   
   	
	
with col4:	
   X15=st.slider("Cu (wt. %)", 0.00, 0.54)	
   X16 = st.slider("Ta (wt. %)", 0.00, 0.36)
   X17= st.slider("Al (wt. %)", 0.00, 100.00)
   X18 = st.slider("V (wt. %)", 0.00, 2.00)
  	

with col5:	
   X19 = st.slider("Mg (wt. %)", 0.00, 1.63)
   X20= st.slider("W (wt. %)", 0.00, 2.40)
	
   X21 = st.slider("Zr (wt. %)", 0.00, 0.10)  
   X22=st.slider("Dose rate /1000 (dpa/s)", 0.000, 60.00)

	
with col6:	
   X23 = st.slider("Temperature (K)", 393.5, 1013.5)
   X24= st.slider("Irradiation type [1-4]", 1, 4)
   X25 = st.slider("Dislocation density x10^14 (m−2)", 0.30, 38.50)
   X26 = st.slider("Pre-injected He (appm)", 0.00, 100.00)


from sklearn.model_selection import train_test_split, KFold
#from sklearn.ensemble import GradientBoostingRegressor
#import catboost as cb
from catboost import CatBoostRegressor

#Model

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.read_csv('Swelling_data_Thanh.csv')
X_ori = df[['Fe','Cr','Mn','Si','Co','Mo','C','Ti','B','P','S','Nb','Cu','Ta','Al','V','Mg','W','Zr','dr','tem','irrtype','disden','PreHe']].values
y = df['dose'].values
X = scaler.fit_transform(X_ori)	

Inputdata = [X1, X2, X3, X4, X5/100, X6,X8, X9, X11/1000, X12/100, X13/100, X14, X15, X16, X17,X18, X19, X20, X21, X22/1000, X23, X24, X25*(10**14),X26]
predictions = []
from numpy import asarray
X_predict = asarray([Inputdata])
  
for i in range(1,6,1):
    count = i
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=(0+i))
    
    n_estimators = 452 #430 #120
    learning_rate = 0.301 #0.513
    depth = 7
    l2_leaf_reg = 6
    
    cat_clf_n = CatBoostRegressor(n_estimators=n_estimators,learning_rate = learning_rate, depth=depth,l2_leaf_reg=l2_leaf_reg, random_state=42)
    cat_clf_n.fit(X_train, y_train)


    # X_predict1= scaler.fit_transform(X_predict)
    X_predict1=scaler.transform(X_predict)

    result_xgb = cat_clf_n.predict(X_predict1)

    predictions.append(result_xgb)


mean_pred = np.mean(predictions)
std_pred = np.std(predictions)

puxgb = np.around(mean_pred, 3)
puxgb_err = np.around(std_pred, 3)


st.subheader("Output variable")
if st.button("Predict"):
    import streamlit as st
    import time
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(f"Your predicted incubation dose (dpa) obtained from HHO-CGB model is: {(puxgb)}")
    st.success(f"The standard deviation (Uncertainty value) (dpa) is: {(puxgb_err)}")
