#!/usr/bin/env python
# coding: utf-8

# # 1 Occupations 

# ### 1.a Import the dataset 

# In[1]:


import pandas as pd


# In[2]:


FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"


# In[3]:


df=pd.read_csv(FNAME, delimiter ='|')


# In[4]:


df.head(7)


# ### 1.b Print the last 10 entries and the first 25 entries

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


df.tail(10)


# In[8]:


df.head(25)


# ###  1.c Type of each column of the dataset 

# In[ ]:





# ### 1.d Age values with the least occurrence 

# In[9]:


df["age"].value_counts()


# ### 1.e Frecuency distribution occupation 

# In[10]:


df["occupation"].value_counts()


# ### 1.f Type of the object created 

# ********

# ### 1.g Frecuency distribution occupation type 

# In[11]:


df["occupation"].value_counts()


# ### 1.h Histogram for occupation 

# In[17]:


df['occupation'].value_counts().plot(kind='bar')
plt.title('types of occupation')
plt.xlabel ('types', fontsize = "medium")
plt.ylabel('frecuency')
plt.xticks(rotation='vertical')
plt.savefig('occupation_types.pdf')

