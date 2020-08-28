#!/usr/bin/env python
# coding: utf-8

# ### 1.a Load dataset 

import pandas as pd

import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()

data = pd.DataFrame(boston.data, columns=boston.feature_names)
data.head()


# #### Boston contains 506 instances/rows and 13 predictive variables (numeric/categorical)

data['MEDV'] = pd.Series(data=boston.target, index=data.index)

data.describe()


# ### 1.b Extract polynomial features and interactions up to a degree of 2 


X = data.drop('MEDV', axis = 1)
Y = data['MEDV']


# #####  By default train_test_split divide the sample assigning 25% for train test group 

from sklearn.preprocessing import PolynomialFeatures


poly = PolynomialFeatures(degree=2,include_bias=False).fit(X)
X_poly = poly.transform(X)
print("X_poly.shape: {}".format(X_poly.shape))


# ##### The polynominal transformation dataset includes 104 features, 13 original features, new 13 squared values of the original and 78 interaction among these variables. The inclusion of these new variables should improve model fitting from a lineal version which do not take into account non-lineal relation among features (explanatory variables) and dependent variable (av. price of Boston houses or MEDV)

print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


# #####  The polynominal transformation dataset includes 105 features, 13 original features, new 13 squared values of the original and 70 interaction among these variables. The inclusion of these new variables should improve model fitting from a lineal version which do not take into account non-lineal relation among features (explanatory variables) and dependent variable (av. price of Boston houses)  

# ### 1.c Create a pandas DataFrame using the polynomials and save the file 


polynomials0 = pd.DataFrame(X_poly, columns =['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x0 x4', 'x0 x5', 'x0 x6', 'x0 x7', 'x0 x8', 'x0 x9', 'x0 x10', 'x0 x11', 'x0 x12', 'x1^2', 'x1 x2', 'x1 x3', 'x1 x4', 'x1 x5', 'x1 x6', 'x1 x7', 'x1 x8', 'x1 x9', 'x1 x10', 'x1 x11', 'x1 x12', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x2 x6', 'x2 x7', 'x2 x8', 'x2 x9', 'x2 x10', 'x2 x11', 'x2 x12', 'x3^2', 'x3 x4', 'x3 x5', 'x3 x6', 'x3 x7', 'x3 x8', 'x3 x9', 'x3 x10', 'x3 x11', 'x3 x12', 'x4^2', 'x4 x5', 'x4 x6', 'x4 x7', 'x4 x8', 'x4 x9', 'x4 x10', 'x4 x11', 'x4 x12', 'x5^2', 'x5 x6', 'x5 x7', 'x5 x8', 'x5 x9', 'x5 x10', 'x5 x11', 'x5 x12', 'x6^2', 'x6 x7', 'x6 x8', 'x6 x9', 'x6 x10', 'x6 x11', 'x6 x12', 'x7^2', 'x7 x8', 'x7 x9', 'x7 x10', 'x7 x11', 'x7 x12', 'x8^2', 'x8 x9', 'x8 x10', 'x8 x11', 'x8 x12', 'x9^2', 'x9 x10', 'x9 x11', 'x9 x12', 'x10^2', 'x10 x11', 'x10 x12', 'x11^2', 'x11 x12', 'x12^2'])


polynomials0.head()


polynomials0 = polynomials0.join(Y)


polynomials0.head()

polynomials= polynomials0.rename(columns= {"MEDV":"Y"})

polynomials.head()

polynomials.shape

polynomials.to_csv(r'C:\Users\JULIO\Documents\GitHub\proyectASP2020\ex_machine_learning_1_5\1_fet_eng\polynomials.csv')

