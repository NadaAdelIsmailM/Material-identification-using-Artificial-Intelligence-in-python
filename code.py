from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd #To deal with data
from sklearn.linear_model import LinearRegression #To make the prediction model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

dataset =  pd.read_csv('linear updated.csv') #DataFrame object 
sample = pd.read_csv('sample2.csv')

y_train = dataset['X']
X_train = dataset.drop('X', axis=1)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_encoded

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train_encoded)
y_pred = regr.predict(sample)
y_pred = y_pred.round().astype(int)


label_encoder.inverse_transform(y_pred)