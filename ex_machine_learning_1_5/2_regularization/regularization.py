#!/usr/bin/env python
# coding: utf-8

# ### 2.a Read the data from exercise "Feature Engineering" 

import pandas as pd

import numpy as np

FNAME = r"C:\Users\JULIO\Documents\GitHub\proyectASP2020\ex_machine_learning_1_5\1_fet_eng\polynomials.csv"

data = pd.read_csv(FNAME, delimiter =',')


# ### 2.b Split variables Y and X 


from sklearn.datasets import load_boston


from sklearn.model_selection import train_test_split


X = data.drop('Y', axis = 1)
Y = data['Y']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), np.arange(2, 5, 0.5), np.arange(5, 25, 1)))}
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### 2.c Learn a Ridge model and a Lasso model using the provided data with default parameters 

# #### Ridge model 

from sklearn.linear_model import Ridge


from sklearn import linear_model


from sklearn.metrics import mean_squared_error


from sklearn.model_selection import GridSearchCV


ridge = linear_model.Ridge()

gridridge = GridSearchCV(ridge, parameters, scoring ='r2')


gridridge.fit(X_train, Y_train)


# #### Lasso model 

from sklearn.linear_model import Lasso


lasso = linear_model.Lasso()


gridlasso = GridSearchCV(lasso, parameters, scoring ='r2')


gridlasso.fit(X_train, Y_train)


print("lasso best parameters:", gridlasso.best_params_)
print("lasso score:", gridlasso.score(X_test, Y_test))
print("lasso MSE:", mean_squared_error(Y_test, gridlasso.predict(X_test)))
print("lasso best estimator coef:", gridlasso.best_estimator_.coef_)


# ### 2.d Create a DataFrame containing the learned coefficients of both models and the feature names as index


# ### 2.e Using matplotlib.pyplot, create a horizontal bar plot of dimension 10x30 showing the coefficient sizes 

import matplotlib.pyplot as plt

coefsLasso = []
coefsRidge = []
alphasLasso = np.arange (0, 20, 0.1)
alphasRidge = np.arange (0, 200, 1)
for i in range(200):
    lasso = linear_model.Lasso(alpha=alphasLasso[i])
    lasso.fit(X_train, Y_train)
    coefsLasso.append(lasso.coef_)
    ridge = linear_model.Ridge(alpha=alphasRidge[i])
    ridge.fit(X_train, Y_train)
    coefsRidge.append(ridge.coef_[0])


plt.figure(figsize = (30,10))
plt.subplot(121)
plt.plot(alphasLasso, coefsLasso)
plt.title('Lasso coefficients')
plt.xlabel('alpha')
plt.ylabel('coefs')
plt.subplot(122)
plt.plot(alphasRidge, coefsRidge)
plt.title('Ridge coefficients')
plt.xlabel('alpha')
plt.ylabel('coefs')
plt.show()


fig = plt.figure()
plt.plot(alphasLasso, coefsLasso)
fig.suptitle('alphas Lasso', fontsize=15)
fig.savefig('lasso.pdf')


fig = plt.figure()
plt.plot(alphasRidge, coefsRidge)
fig.suptitle('alphas Ridge', fontsize=15)
fig.savefig('Ridge.pdf')

