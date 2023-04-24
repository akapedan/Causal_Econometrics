"""
Data Analytics II: PC3 Functions.

Spring Semester 2022.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np

# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings, unique values and number of observations
def my_summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
        
    Returns
    -------
    None. Prints descriptive table od the data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')


# write a function to compute sum of squared errors (SSE)
def my_sse(prediction):
    """
    Compute SSE given prediction
    
    Parameters
    ----------
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
        
    Returns
    -------
    result: SSE
    
    """
    # compute sse using formula
    sse = ((prediction - prediction.mean()) ** 2).sum()
    return sse

# write a function finding the best split using sse
def my_best_split(covariates, outcome, minleaf=10):
    """
    Find optimal split by minimzing sse
    
    Parameters
    ----------
    covariates : TYPE: pd.Series
        DESCRIPTION: vector of covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    minleaf : TYPE: integer
        DESCRIPTION: minimum leaf size for a tree 
        
        
    Returns
    -------
    result: optimal splitting value, sse and index.
    
    """
    
    # create pandas dataframe
    df = pd.DataFrame(pd.concat([outcome, covariates], axis=1), 
                      columns=['Y', 'X'])
    # number of observations
    N = df.shape[0]
    
    # create empty storage for sse for daughter nodes
    sse_left = []
    sse_right = []    
    
    # splitting grid
    grid = np.arange(minleaf, N - minleaf, 1)
    
    # loop over all possible split points given grid
    for split in grid:
        sse_left.append(my_sse(df.Y.iloc[0:split]))
        sse_right.append(my_sse(df.Y.iloc[split:N]))
        
    # global sse
    sse_global = pd.Series(np.add(sse_left, sse_right), index=grid)

    # find minimum sse
    optimal_sse = np.min(sse_global)
    
    # optimal index location in sample
    optimal_index_sample = sse_global.index[sse_global == optimal_sse]
    
    # optimal value
    optimal_value = df['X'].iloc[optimal_index_sample]
    
    # optimal index
    optimal_index = optimal_value.index[0]
    
    # results
    result = {'sse': optimal_sse, 'value': float(optimal_value), 'index': optimal_index}
    
    return result

# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""
