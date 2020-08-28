#!/usr/bin/env python
# coding: utf-8

# ### 3.a import dataset from sklearn.datasets 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn.neural_network as ml

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()


data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data.head(7)

data.shape
list(data)


# #### diabetes dataset contains 442 observations, 10 features (explanatory variables) and 1 target/dependent variable (disease progression) 

data['dia'] = pd.Series(data=diabetes.target)
data.head()


pd.set_option('float_format', '{:f}'.format)
print(data.describe())


# #### data of the 10 features have been  scaled (mean of the scaled values equal to zero) but not the values of the quantitative measure of desease (dia) 

data['dia'].describe()


# #### rescale the dataset, assuming X´s features and output (Y) mean equal to 0 and standard deviation equal to 1 


from sklearn.preprocessing import StandardScaler


scaling = StandardScaler()


scaling.fit_transform(data)


data_sc = pd.DataFrame(scaling.fit_transform(data), columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'Y'])



data_sc.describe()


# ### 3.b Learn a a Neural Network with 1000 iterations 


X = data_sc.drop('Y', axis = 1)
Y = data_sc['Y']


from sklearn.model_selection import train_test_split


from sklearn.model_selection import KFold


kf = KFold(n_splits= 4)
X = np.array(X)
Y = np.array(Y)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


# #### K-fold cross validation defining 4 samples or folds 


from sklearn.neural_network import MLPRegressor


regr = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='tanh', solver='lbfgs', random_state=1, max_iter= 1000).fit(X_train, Y_train)


X_test.shape, Y_test.shape

X_train.shape, Y_train.shape


# ### 3.c  Best parameters selection 




