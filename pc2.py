"""
Data Analytics II: PC2.

Spring Semester 2022.

University of St. Gallen.
"""

# Data Analytics II: PC Project 2

# import modules here
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# set working directory
path = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC2/'
sys.path.append(path)

# load own functions
import pc2_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc2_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=path, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc2.csv'

# load in data using pandas
data = pd.read_csv(path + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
## Part 1:
# --------------------------------------------------------------------------- #
## Exercise 1b)
# call Descriptive Statistics
pc.my_adjstats(data=data)
# call all Histograms
pc.my_hist(data=data, path=path, save=True)

## Exercise 1c)
# remove msmoke and monthslb from df
data = data.drop(['msmoke', 'monthslb'], axis=1)
# remove remaining rows with missing values 
data = data.dropna(axis=0)

## Exercise 1d) 
# recode order variable by applying function 
# coding: 1=first infant birth, 0=else
data['order'] = data['order'].apply(pc.order_recode)
# recode prenatal variable by applying function 
# coding: 1=first prenatal visit in first trisemester, 0=else
data['prenatal'] = data['prenatal'].apply(pc.prenatal_recode)

## Exercise 1e)
# call Descriptive Statistics 
pc.my_adjstats(data=data)

# save clean df in new csv
data.to_csv(path + 'data_pc2_clean.csv')

## Exercise 1f) 
pc.balance_check(data=data, treatment='mbsmoke', variables=('bweight', 'mhisp', 'alcohol', 'deadkids', 'mage', 'medu', 'nprenatal', 'order', 'mrace', 'prenatal'))
# --------------------------------------------------------------------------- #
## Part 2:
# --------------------------------------------------------------------------- #
## Exercise 2a)
# define dependent and independent variable
Y=data['bweight']
X=data['mbsmoke']
# add constant
X=sm.add_constant(X)
# perform OLS-regression
reg=sm.OLS(Y, X)
res=reg.fit()
# print results of OLS-regression
print(res.summary())

## Exercise 2b)
# define dependent and independent variable
Y=data['bweight']
X=data[['mbsmoke', 'mhisp', 'alcohol', 'deadkids', 'mage', 'medu', 'nprenatal', 'order', 'mrace', 'prenatal']]
# add constant
X=sm.add_constant(X)
# perform OLS-regression
reg=sm.OLS(Y, X)
res=reg.fit()
# print results of OLS-regression
print(res.summary())
# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 2 Session #
