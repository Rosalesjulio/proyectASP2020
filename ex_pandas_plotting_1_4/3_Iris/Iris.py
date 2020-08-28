#!/usr/bin/env python
# coding: utf-8

# ### 3.a Upload dataset


import pandas as pd


df= pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', sep = ',')


df.head()


# ### adding names of columns in uploaded dataset 

# #### names to be included in each column: sepal_lenght, sepal_width, petal_length, petal_width, class 


df.columns


df.shape


df= pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', sep = ',', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])


df.head()


# ### 3.b Set the values of the rows 10 to 29 of the column ’petal length (in cm)’ to missing 


df.head(30)


df.loc[10:29,['petal_length']] = "NaN"


df.head(30)


# ### 3.c Replace missing values of rows 10 - 20 with 1.0 


df.loc[10:29,['petal_length']] = 1


df.head(30)


# ### 3.d Save the data as comma-separated file named ./output/iris.csv without index. 


df.to_csv(r'C:\Users\JULIO\Documents\GitHub\proyectASP2020\ex_pandas_plotting_1_4\3_Iris\iris.csv', index = False)


# ### 3.e Visualize the distribution of all of the continuous variables by ”class” with a catplot of your choice. Save the figure 

# #### continuous variables: sepal_l, sepal_w, petal_l, petal_l by class 


df['class'].unique()  ### class has 3 categories: Iris-s, Iris-ver and Iris-virg 


import matplotlib.pyplot as plt


import seaborn as sns


# ####  Sepal_length by class

g = sns.catplot(x="class", y= "sepal_length", data= df)
plt.savefig('sepal_length.pdf')


# #### Sepal_width by class 


g = sns.catplot(x="class", y= "sepal_width", data= df)
plt.savefig('sepal_width.pdf')


# #### Petal_length by class 

g = sns.catplot(x="class", y= "petal_length", data= df)
plt.savefig('petal_length.pdf')


# ##### it is possible to say that there is a relatively high degree of diference in the values of variable petal_length by these tree categories. In terms of dispersion, Petal-length is more concentrated in Iris-Set category  

# #### Petal_width by class 


g = sns.catplot(x="class", y= "sepal_length", data= df)
plt.savefig('petal_width.pdf')

