# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SAKTHIVEL R
RegisterNumber: 212221040141
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

Initial Dataset:

![ml5](https://user-images.githubusercontent.com/93427345/174268804-73804cb8-c5fa-46cd-ac82-11ba9868b984.PNG)

Dataset information:

![ml51](https://user-images.githubusercontent.com/93427345/174268854-9fa6865d-93ba-4ded-b042-3e8ce5b903f4.PNG)

Left column value count:

![ml52](https://user-images.githubusercontent.com/93427345/174268895-d3df1678-2955-47bf-a895-0e6ff4e34ceb.PNG)

Encoded Dataset:

![ml53](https://user-images.githubusercontent.com/93427345/174268919-d2f48fc5-6988-4e56-b0cb-7bdfcf1745a9.PNG)

X set:

![ml54](https://user-images.githubusercontent.com/93427345/174268983-0c7e6bd0-5bd2-440d-84a5-8def5126921d.PNG)

Y values:

![ml55](https://user-images.githubusercontent.com/93427345/174269004-ef08308d-c2e2-4d4c-b5b2-824801528021.PNG)

Accuracy score:

![ml](https://user-images.githubusercontent.com/93427345/174269041-0150acb4-b614-4ded-b46b-358361e144bb.PNG)

Result value of Model when tested:

![ml56](https://user-images.githubusercontent.com/93427345/174269689-45ac8114-0150-4b8c-b3ee-7e57e2623969.PNG)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
