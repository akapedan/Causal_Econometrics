"""
Data Analytics II: PC3.

Spring Semester 2022.

University of St. Gallen.
"""

# Data Analytics II: PC Project 3

# import modules here
import sys
import pandas as pd
import numpy as np


# set working directory
PATH = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC3/'
sys.path.append(PATH)

# load own functions
import pc3_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc3_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc3.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# set seed
np.random.seed(123)

# your solutions start here
# --------------------------------------------------------------------------- #
# Exercise 1a)
pc.my_summary_stats(data)

# Exercise 1b)
X = data[['X']]
Y = data[['Y']]

# compute sse using function
sse = pc.my_sse(Y)

# store number of observations in N
N = data.shape[0]

# compute mse
mse = sse / N

print('SSE : ', round(float(sse),4), '\n', 'MSE : ', round(float(mse), 4), '\n')

# Exercise 1c)
# sort the data in an increasing order according to X and reset the indices 
data = pd.DataFrame(data.sort_values(by='X')).reset_index(drop=True)

# set minimum leaf size
minleaf = 10

# first split
first_split = pc.my_best_split(X, Y, minleaf)


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 3 Session #
