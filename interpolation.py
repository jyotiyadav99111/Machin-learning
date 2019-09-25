#!/usr/bin/env python
# coding: utf-8

# In[84]:

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tools.eval_measures import rmse

df = pd.read_csv('https://raw.githubusercontent.com/jyotiyadav99111/Time-Series/master/web_traffic.csv')

df.head()

# Converting column to the right format and setting the index to the dataframe
df.Date = pd.to_datetime(df.Date)
df['Web Traffic'] = pd.to_numeric(df['Web Traffic']) 
df = df.set_index('Date')
df.head()

# Visualisation of Data
df.plot(style = ['k--', 'bo-', 'r*'], figsize = (20,7))


# Trying different methods for imputation

# Using mean and median

df = df.assign(mean = df.Missing.fillna(df.Missing.mean()))
df = df.assign(median = df.Missing.fillna(df.Missing.median()))


# Using rolling average

df = df.assign(RollMean = df.Missing.fillna(df.Missing.rolling(24, min_periods = 1).mean()))
df = df.assign(RollMedian = df.Missing.fillna(df.Missing.rolling(24, min_periods = 1).median()))


# Imputation using interpolation from sklearn with different methods

df = df.assign(Int_Time = df.Missing.interpolate(method = 'time'))
df = df.assign(Int_Akima = df.Missing.interpolate(method = 'akima'))
df = df.assign(Int_Cubic = df.Missing.interpolate(method = 'cubic'))
df = df.assign(int_Linear = df.Missing.interpolate(method = 'linear'))
df = df.assign(Int_Slinear = df.Missing.interpolate(method = 'slinear'))
df = df.assign(Int_Quadratic = df.Missing.interpolate(method = 'quadratic'))
df = df.assign(Int_Spline3 = df.Missing.interpolate(method = 'spline', order = 3))
df = df.assign(Int_Spline4 = df.Missing.interpolate(method = 'spline', order = 4))
df = df.assign(Int_Spline5 = df.Missing.interpolate(method = 'spline', order = 5))
df = df.assign(Int_Ploy5 = df.Missing.interpolate(method = 'polynomial', order = 5))
df = df.assign(Int_Ploy7 = df.Missing.interpolate(method = 'polynomial', order = 7))

# Calculating metrices to compare the imputation method

results = [(method, r2_score(df['Web Traffic'], df[method]), rmse(df['Web Traffic'], df[method], axis = 0)) for method in list(df)[3:]]

results_df = pd.DataFrame(np.array(results), columns = ['Method', 'R_squared', 'RMSE'])

results_df.sort_values(by = 'R_squared', ascending = False)


# Visualisation the results
df_plot = df.iloc[0:100, :]
df_plot.plot(style = ['k--', 'bo-', 'r*'], figsize = (20,7))


