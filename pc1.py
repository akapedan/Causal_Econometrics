"""
Data Analytics II: PC1.

Spring Semester 2022.

University of St. Gallen.
"""

# Data Analytics II: PC Project 1

# import modules here
import sys
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# set working directory
PATH = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC1/'
sys.path.append(PATH)

# load own functions
import pc1_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc1_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc1.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
## Exercise 1b)
print(data.describe())

## Exercise 1c)
pc.my_adjstats(data=data)

## Exercise 1d)
data = data.drop(['age2'], axis=1)

## Exercise 1e)
pc.my_hist(data=data, save=True)

## Exercise 1f)
data.to_csv(PATH + 'final_df.csv')

## Exercise 1g)
pc.balance_check(data=data, treatment='treat', variables=('age', 'ed', 'black', 'hisp', 'married', 'nodeg', 're74', 're75'))

## Exercise 2a)
pc.ate_md(outcome=data['re78'], treatment=data['treat'])

## Exercise 2b)
# define dependent and independent variable
Y=data['re78']
X=data['treat']
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

# End of the PC 1 Session #
