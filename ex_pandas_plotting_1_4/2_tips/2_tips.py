#!/usr/bin/env python
# coding: utf-8

# ### 2.a Load dataset 

import pandas as pd


FNAME = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"


df=pd.read_csv(FNAME, delimiter =',')


df.head(7)


# ### 2.b Replace short names for full names- variable day using function replace() 

df['day'].value_counts()


df['day'].unique()


# #### Dataset includes data collected only for the last 4 days of the week 

df['day'].replace(('Sun', 'Sat', 'Thur', 'Fri'), ('Sunday', 'Saturday', 'Thrusday', 'Friday'),inplace=True)

df['day'].unique()


# ### 2.c Scatter plot of "tips" vs "total_Bill" by gender 

import matplotlib.pyplot as plt

import seaborn as sns

g = sns.lmplot(x="total_bill", y="tip",hue= "day", col= "sex", fit_reg= False, data= df)
plt.savefig('scatter_plot_tips_bill_bysex.pdf')

