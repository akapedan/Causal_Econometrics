"""
Data Analytics II: PC4.

Spring Semester 2022.

University of St. Gallen.
"""

# Data Analytics II: PC Project 4

# import modules
import sys
import pandas as pd

# set working directory
PATH = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC4/'
sys.path.append(PATH)

# load own functions
import pc4_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc4_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc4.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
# Ex. 1b)
# check descriptives
pc.my_adjstats(data)  

# Ex. 1c)
# plot distribution of the outcome fte for each state and year
# loop over unique values in treatment
for state in data['state'].unique():
    # loop over unique values in time
    for year in data['year'].unique():
        # create subset
        df = pd.DataFrame(data.loc[(data['state'] == state) &
                                        (data['year'] == year), 'fte'])
        # plot the histogram
        pc.my_hist(data=df, varname='fte', path=PATH, nbins=20,
                   label=('Pennsylvania in \n'+'19'+str(year) if state == 0 else 'New Jersey in \n'+'19'+str(year)))
        
# plot distribution of change of the outcome fte across states and years
# loop over unique values in treatment
for state in data['state'].unique():
    outcome93 = pd.DataFrame(data.loc[(data['state']==state) 
                                      & (data['year']==93), 
                                      'fte']).reset_index(drop=True)
    outcome92 = pd.DataFrame(data.loc[(data['state']==state) 
                                      & (data['year']==92), 
                                      'fte']).reset_index(drop=True)
    outcome_diff = outcome93 - outcome92
    pc.my_hist(outcome_diff, 'fte', PATH, 20, label=('Pennsylvania time difference' if state == 0 else 'New Jersey time difference'))

# Ex. 1d)
# check if regional dummies southj, centralj, northj, pa1, and pa2 are well defined
# new jersey
dummy_nj = ('northj', 'southj', 'centralj')
pc.dummy_check(data, dummies=dummy_nj, name=('New Jersey', ))
    
# pennsylvania
dummy_pa = ('pa1', 'pa2')
pc.dummy_check(data, dummies=dummy_pa, name=('Pennsylvania', ))

# Ex. 1e)
# mean table of fte, wage_st, hrsopen, price for each state and year
x_name = ('wage_st', 'hrsopen', 'price')
pc.mean_table(data, byvar=('state', 'year'), meanvar=('fte', ) + x_name)

# Ex. 1f)
# create dummies for the fastfood chain and remove the original variable chain
dummies_chain = pd.get_dummies(data.loc[:, 'chain'], prefix='chain')
# add dummies to the data and remove the original variable
data = pd.concat([data.loc[:, data.columns != 'chain'], dummies_chain],
                 axis=1)

# recode year variable into dummy
data['year'] = (data['year'] == 93) * 1

# check descriptives again
pc.my_adjstats(data)

#Ex. 2a)
# estimate the effect of higher minimum wages on full time equivalent
# employment in fast food restaurants by mean difference between New Jersey 
# and Pennsylvania after the policy change
ate_state = pc.ate_md(outcome=data.loc[data['year'] == 1, 'fte'],
                      treatment=data.loc[data['year'] == 1, 'state'],
                      display=True)

#Ex. 2b)
# estimate the effect ofhigher minimum wages on full time equivalent 
# employment in fast food restaurants by mean difference in New Jersey 
# before and after the policy change
ate_nj_time = pc.ate_md(outcome=data.loc[data['state'] == 1, 'fte'],
                     treatment=data.loc[data['state'] == 1, 'year'],
                     display=True)
# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 4 Session #
