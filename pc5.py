"""
Data Analytics II: PC5.

Spring Semester 2022.

University of St. Gallen.
"""

# Data Analytics II: PC Project 5

# import modules
import sys
import pandas as pd
import numpy as np

# set working directory
PATH = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC5/'
sys.path.append(PATH)

# load own functions
import pc5_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc5_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc5.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
## Part 1
# Ex. 1b)
pc.my_adjstats(data)

pc.my_hist(data=data, varname='kidcount', path=PATH,nbins=10)
pc.my_hist(data=data, varname='weeks_work', path=PATH,nbins=10)

# Ex. 1c)
pc.employed_kidcount(data=data, varname='employed', groupname='kidcount')

# Ex. 1d)
pc.crosstable(data=data, var1='morekids', var2='multi2nd')

## Part 2
# Exercise 2a)
pc.my_ols(exog=data[['morekids', 'age_mother', 'black', 'hisp', 'hsgrad', 'colgrad']],
          outcome=data['weeks_work'],
          instrument=0,
          endogname=0,
          intercept=True, display=True, SLS=False)

### Exercise 2b)
# 2SLS
pc.my_ols(exog=data[['morekids', 'age_mother', 'black', 'hisp', 'hsgrad',
                     'colgrad']],
          outcome=data['weeks_work'],
          intercept=True, display=True, SLS=True, 
          instrument=data['multi2nd'], 
          endogname='morekids')

# Assumption 2 test
pc.my_ols(exog=data[['multi2nd', 'age_mother', 'black', 'hisp', 'hsgrad', 'colgrad']],
          outcome=data['morekids'],
          instrument=0,
          endogname=0,
          intercept=True, display=True, SLS=False)

# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 5 Session #
