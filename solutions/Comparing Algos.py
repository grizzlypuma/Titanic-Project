import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
import time

X_train_data = pd.read_csv("C:/Users/kazja/OneDrive/Documents/AlphaQuant/X_train_titanic.csv")
y_train_data = pd.read_csv("C:/Users/kazja/OneDrive/Documents/AlphaQuant/y_train_titanic.csv")

print(X_train_data.head())
print(y_train_data.head())

X_train, x_val, y_train, y_val = train_test_split(X_train_data, y_train_data, test_size=0.3)
X_train = X_train.values
x_val = x_val.values
y_train = y_train.values
y_val = y_val.values


ada_clf = AdaBoostClassifier(random_state=42)

start_time = time.time()
ada_clf.fit(X_train, y_train)
end_time = time.time()  # Get the end time after the code execution
execution_time_ada = end_time - start_time

print( f"Execution time for AdaBoost is {execution_time_ada:.6f}")

prediction_train_ada = ada_clf.predict(X_train)
accuracy_train = accuracy_score(prediction_train_ada, y_train)

prediction_val_ada = ada_clf.predict(x_val)
accuracy_val = accuracy_score(prediction_val_ada, y_val)

print(f"Model prediction score: {accuracy_train:.2f}")
print(f"Model validation score: {accuracy_val:.2f}")

print("\n")
print("//////////////////////////////////////////////")
print("\n")


rf_clf = RandomForestClassifier(random_state=42)

start_time = time.time()
rf_clf.fit(X_train, y_train)
end_time = time.time()  # Get the end time after the code execution
execution_time_rf = end_time - start_time

print( f"Execution time for Random Forrest is {execution_time_rf:.6f}")

prediction_train_rf = rf_clf.predict(X_train)
accuracy_train = accuracy_score(prediction_train_rf, y_train)

prediction_val_rf = rf_clf.predict(x_val)
accuracy_val = accuracy_score(prediction_val_rf, y_val)

print(f"Model prediction score: {accuracy_train:.2f}")
print(f"Model validation score: {accuracy_val:.2f}")

print("\n")
print("//////////////////////////////////////////////")
print("\n")

knn = KNeighborsClassifier()

start_time = time.time()
knn.fit(X_train, y_train)
end_time = time.time()
execution_time_knn = end_time-start_time

print( f"Execution time for KNN is {execution_time_knn:.6f}")

prediction_train_knn = knn.predict(X_train)
accuracy_train = accuracy_score(prediction_train_knn, y_train)

prediction_val_knn = knn.predict(x_val)
accuracy_val = accuracy_score(prediction_val_knn, y_val)

print(f"Model prediction score: {accuracy_train:.2f}")
print(f"Model validation score: {accuracy_val:.2f}")



