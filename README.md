# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: D. Amarnath Reddy
RegisterNumber: 212221240012  
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```

## Output:
<img width="647" alt="1" src="https://user-images.githubusercontent.com/94165103/162415466-68ca5873-8111-4e42-94af-28fdbd542896.png">
<img width="648" alt="Screenshot 2022-04-08 at 3 32 33 PM" src="https://user-images.githubusercontent.com/94165103/162415483-526af12e-915b-4897-9a01-de3d3672cb52.png">
<img width="647" alt="2" src="https://user-images.githubusercontent.com/94165103/162415504-09a0449a-255f-4623-b231-3bad97f6271b.png">
<img width="647" alt="3" src="https://user-images.githubusercontent.com/94165103/162415514-7cb845d8-04c7-423d-80fb-f0034da357b6.png">
<img width="647" alt="4" src="https://user-images.githubusercontent.com/94165103/162415524-efd8c874-dfcc-4c7e-9e6d-b0e164acb214.png">
<img width="647" alt="5" src="https://user-images.githubusercontent.com/94165103/162415535-5e43577f-ea61-4229-95b5-47ff616d771f.png">
<img width="647" alt="6" src="https://user-images.githubusercontent.com/94165103/162415555-47331a02-7e0a-4120-83c3-efefe64db032.png">
<img width="647" alt="7" src="https://user-images.githubusercontent.com/94165103/162415561-f1d29e7e-1e65-4532-b9f2-a40911ea92de.png">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
