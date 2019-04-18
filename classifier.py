#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Random Forest Classification
# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/siddharth/NHC.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 0].values

#only require to convert this column for the equation
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y.astype(str))
y = y.astype(float)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#dataset = dataset.reset_index()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import pickle
with open('storm_rf_classifier.pkl', 'wb') as file:
	pickle.dump(classifier, file)










