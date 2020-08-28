#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer = load_breast_cancer()


# In[7]:


data = pd.DataFrame(cancer.data, columns= cancer.feature_names)
data.head(7)


# In[8]:


list(data.columns)


# #### breast cancer dataset contains 569 observations, 30 features (explanatory variables) 
# 

# In[12]:


data['Y'] = pd.Series(data=cancer.target) ### the target value of the dataset is 'diagnosis'
data.head(20)


# In[11]:


data.shape


# #### the output/ dependend variable is "diagnosis" and is a categorical one, asuming values of 0 and 1 (=0 malignant, =1 benign)

# ### 5.c Create a histogram for y

# In[13]:


data["Y"].value_counts()


# In[20]:


label = ['benign', 'malignant']
data['Y'].value_counts().plot(kind='bar')
plt.title('=1 malignant; =0 benign')
plt.xlabel ('diagnosis', fontsize = "medium")
plt.ylabel('frecuency')


# #### the distribution of target variable is according a categorical variable with 2 categories/features 

# ### 5.d Split the data as usual into a test and training set 

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X = data.drop('Y', axis = 1)
Y = data['Y']


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), np.arange(2, 5, 0.5), np.arange(5, 25, 1)))}
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:




