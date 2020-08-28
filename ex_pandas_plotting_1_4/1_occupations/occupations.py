#!/usr/bin/env python
# coding: utf-8

# # 1 Occupations 

# ### 1.a Import the dataset 

import pandas as pd


FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"


df=pd.read_csv(FNAME, delimiter ='|')


df.head(7)


# ### 1.b Print the last 10 entries and the first 25 entries


import matplotlib.pyplot as plt


df.tail(10)


df.head(25)


df.describe()


# ###  1.c Type of each column of the dataset 


df.dtypes


# ### 1.d Age values with the least occurrence 


df["age"].value_counts()


# ### 1.e Frecuency distribution occupation 


df["occupation"].value_counts()


# In[28]:


occup_freq = df["occupation"].value_counts()


occup_freq.head()


# ### 1.f Type of the object created 

# #### the object created by = operator in this case is data labeled with the different categories of occupation 

# ### 1.g Frecuency distribution occupation type 


df["occupation"].value_counts()


# ### 1.h Histogram for occupation 


df['occupation'].value_counts().plot(kind='bar')
plt.title('types of occupation')
plt.xlabel ('types', fontsize = "medium")
plt.ylabel('frecuency')
plt.xticks(rotation='vertical')
plt.savefig('occupation_types.pdf')



