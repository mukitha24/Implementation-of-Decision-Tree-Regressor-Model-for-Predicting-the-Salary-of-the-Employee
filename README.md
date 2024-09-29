# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries .

2.Read the data frame using pandas.

3.Get the information regarding the null values present in the dataframe.

4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

5.Determine training and test data set.

6.Apply decision tree regression on to the dataframe.

7.Get the values of Mean square error, r2 and data prediction. 
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MUKITHA V M
RegisterNumber:212223040119  
*/
```
```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data head:
![169694235-41a469cc-ff3e-4c56-b36c-029319ef1f94](https://github.com/user-attachments/assets/fe635ce8-b58d-4779-8715-649d627b4596)
## DATA INFO:
![169694238-85077655-4a64-4334-b451-997c7ea1937d](https://github.com/user-attachments/assets/3336aa7c-eb58-4c3f-bccf-aba6611f9356)
## Data Head after applying LabelEncoder():
![169694242-dd7cae7b-50db-4864-96aa-ca8eb07514e3](https://github.com/user-attachments/assets/550b6b93-d54f-4bd8-8abb-bdc9436a1afc)

## MSE:
![169694248-eefed989-8fc7-4e80-b3af-992667d1936a](https://github.com/user-attachments/assets/1012c10e-77f8-40ae-b328-e4a9c437d64b)
## r2:
![169694252-b17fc5dd-22fd-46e0-b8de-991fd12528ed](https://github.com/user-attachments/assets/a915bbda-5ea3-4a0d-9c38-17a9122c5c75)
## Data Prediction:
![169694255-16669af0-0ed0-416e-b387-d63f2f3e9dc3](https://github.com/user-attachments/assets/d661b2a6-8eaa-4400-ace3-c3403ad9aaf2)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
