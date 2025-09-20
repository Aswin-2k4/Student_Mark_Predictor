import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split      #importing required libs
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import joblib


#loading file
data_file=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\PJT2.0\exam_score(encoded_&_cleaned)final.csv")
data=pd.DataFrame(data_file)


data = data_file.dropna()     #dropping Nan
print("Shape of Data : ",data.shape,"\n")


#Separateing target and features
input=data.drop('exam_score',axis=1).values
output=data['exam_score'].values


#Scaling
scaler=MinMaxScaler()
input_scaled=scaler.fit_transform(input)
joblib.dump(scaler,r'C:\Users\HP\OneDrive\Documents\PJT2.0\scaler.pkl')
print("scaler saved successfully")


#Training and Testing data spliting
in_train,in_test,out_train,out_test=train_test_split(input_scaled,output,test_size=0.2,random_state=10)


#Setting the model and training
model=LinearRegression()
model.fit(in_train,out_train)

#prediction of training set
train_pred=model.predict(in_train)
train_pred=np.clip(train_pred,0,100)
train_pred=np.round(train_pred,2)

#prediction of testing set
test_pred=model.predict(in_test)
test_pred=np.clip(test_pred,0,100)
test_pred=np.round(test_pred,2)

#Calculating metrics
mse_train = mean_squared_error(out_train, train_pred)

mse_test = mean_squared_error(out_test, test_pred)

r2_train = r2_score(out_train, train_pred)

r2_test = r2_score(out_test,test_pred )


print("Mean Squared Error of trained set:", mse_train)
print("R² Score of trained set:", r2_train)
print("\n")
print("MSE of test set",mse_test)
print("R² Score of test set:", r2_test)

joblib.dump(model,r'C:\Users\HP\OneDrive\Documents\PJT2.0\score_model.pkl')
print("model saved")




































