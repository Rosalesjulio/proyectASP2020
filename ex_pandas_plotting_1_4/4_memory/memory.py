#!/usr/bin/env python
# coding: utf-8

# ### 4.a load dataset 

import pandas as pd


FNAME = r"C:\Users\JULIO\Documents\GitHub\proyectASP2020\ex_pandas_plotting_1_4\4_memory\game_logs.csv"


df= pd.read_csv(FNAME, delimiter =',')


df.head()


df.tail(5)

df.shape


# #### This dataset contains 171,907 observation units and 161 variables 

# ### 4.b Inspect the DataFrame using .info() and with .info(memory_usage="deep") 

df.info()


df.info(memory_usage="deep")


# ####  Acording to df.info DataFrame requires 160.0 MB of memory usage. On the other hand, according to df.info("deep") it is required 544.2 MB, 3.4 times more.     

# ### 4.c Create a copy of the object with only columns of type object by using .select_dtypes(include=[object])


df.dtypes


df_object = df.select_dtypes(include=['object'])


print(df_object)


df_object.head()


df_object.to_csv(r'C:\Users\JULIO\Documents\GitHub\proyectASP2020\ex_pandas_plotting_1_4\4_memory\memory_obj.csv', index = False)


# ### 4.c Look at the summary of this object new (using .describe()). Which columns have very few unique values compared to the number of observations?


df_object.describe(include ='all').T


pd.unique(df_object['acquisition_info'])


df_object['acquisition_info'].value_counts()


pd.unique(df_object['h_league'])


df_object['h_league'].value_counts()

