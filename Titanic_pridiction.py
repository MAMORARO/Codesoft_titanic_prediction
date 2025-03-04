import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScalar
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report


data_frame =pd.read_csv("Data_set.csv")


data_frame = data_frame.drop(columns=["Name","Ticket","Cabin"], errors='ignore')


data_frame["Age"].fillna(data_frame["Age"].median(),inplace=True)
data_frame["Embarked"].fillna(data_frame["Embarked"].mode()[0], inplace=True)

encoder = OneHotEncoder( drop='first', sparse=False)
categorical_features= ["Sex","Embarked"]
categorial_encoded = encoder.fit_transform(data_frame[categorical_features])
categorical_data_frame = pd.DataFrame(categorical_encoded, columns=encode.get_feature_names_out())


X = data_frame.drop(columns=["Survived"])
Y =data_frame["Age","Fare"]
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scalar.transform(X_test[numeric_features])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,Y_train)

Y_pred =model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
