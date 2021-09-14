# Importing essential libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from datetime import datetime
import pickle
import statsmodels.api as sm

# Loading the dataset
df = pd.read_csv('UCI_data.csv')


# Model Building
from sklearn.model_selection import train_test_split
X = df[["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]]
y = df[["Machine failure"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Linear Regression Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)


# Creating a pickle file for the classifier
filename = 'UCI-prediction-rfc-model.pkl'
pickle.dump(lr, open(filename, 'wb'))



