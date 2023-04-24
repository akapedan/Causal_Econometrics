"""
Data Analytics II: PC2 Functions.

Spring Semester 2022.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# set working directory
path = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC2/'
sys.path.append(path)

# set pandas printing option to see all columns of the data
pd.set_option('display.max_columns', 100)
# and without breaking the dataframe into several lines
pd.set_option('expand_frame_repr', False)
# justify the headers for pandas dataframes
pd.set_option('display.colheader_justify', 'right')

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


# Exercise 1b) 
# Function including mean, variance, standard deviation, maximum and minimum, 
# the number of missing and unique values and number of observations 
# as well as the variable names in a single object
def my_adjstats(data):
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
    # create empty df with fitting index
    columns = ['Mean','Variance', 'Std', 'Min', 'Max', 'NA', 'Unique values', 'N']
    index = data.columns
    adjstats = pd.DataFrame(index=index, columns=columns)
    # fill empty df with respective values
    adjstats['Mean'] = data.mean()
    adjstats['Variance'] = data.var()
    adjstats['Std'] = data.std()
    adjstats['Max'] = data.max()
    adjstats['Min'] = data.min()
    adjstats['NA'] = data.isnull().sum(axis=0)
    adjstats['Unique values'] = data.nunique()
    adjstats['N'] = data.count()
    # show df
    print('Descriptive Statistics:', '-' * 80,
          round(adjstats, 2), '-' * 80, '\n\n', sep='\n')

# own procedure to plot histograms
def my_hist(data, path, save):
    """
    Plot histograms.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    save : TYPE: boolean
        DESCRIPTION: boolean defining if the hist should be saved or not

    Returns
    -------
    None. Prints and saves histogram.
    """
    # define variables as column names
    variables=data.columns
    # create loop that creates a histogram for each variable
    for name in variables:
        plot=plt.hist(data[name], bins=30)
        plt.title(name)
        # argument save if user wants to save the 
        # histograms in the corresponding directory
        if save == True:
            plt.savefig(path + 'histogram_of_{}.png'.format(name))
        plt.show()

# Exercise 1c)
# Recode order variable
def order_recode(order):
    if order == 1.0:
        return 1
    else: 
        return 0 
# Recode prenatal variable
def prenatal_recode(prenatal):
    if prenatal == 1.0:
        return 1
    else: 
        return 0  

# own procedure for a balance check
def balance_check(data, treatment, variables):
    """
    Check covariate balance.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data on which balancing checks should be conducted
    treatment : TYPE: string
        DESCRIPTION: name of the binary treatment variable
    variables : TYPE: tuple
        DESCRIPTION: names of the variables for balancing checks

    Returns
    -------
    Returns and Prints the Table of Descriptive Balancing Checks
    """
    # create storage for output as an empty dictionary for easy value fill
    balance = {}
    # loop over variables
    for varname in variables:
        # define according to treatment status by logical vector of True/False
        # set treated and control apart using the location for subsetting
        # using the .loc both labels as well as booleans are allowed
        treated = data.loc[data[treatment] == 1, varname]
        control = data.loc[data[treatment] == 0, varname]
        # compute difference in means between treated and control
        mdiff = treated.mean() - control.mean()
        # compute the corresponding standard deviation of the difference
        mdiff_std = (np.sqrt(treated.var() / len(treated)
                     + control.var() / len(control)))
        # compute the t-value for the difference
        mdiff_tval = mdiff / mdiff_std
        # get the degrees of freedom (unequal variances, Welch t-test)
        d_f = (mdiff_std**4 /
               (((treated.var()**2) / ((len(treated)**2)*(len(treated) - 1))) +
                ((control.var()**2) / ((len(control)**2)*(len(control) - 1)))))
        # compute pvalues based on the students t-distribution (requires scipy)
        # sf stands for the survival function (also defined as 1 - cdf)
        mdiff_pval = stats.t.sf(np.abs(mdiff_tval), d_f) * 2
        # compute the standardized difference
        sdiff = (mdiff / np.sqrt((treated.var() + control.var()) / 2)) * 100
        # combine values
        balance[varname] = [treated.mean(), control.mean(),
                            mdiff, mdiff_std, mdiff_tval, mdiff_pval, sdiff]
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    balance = pd.DataFrame(balance,
                           index=["Treated", "Control", "MeanDiff", "Std",
                                  "tVal", "pVal", "StdDiff"]).transpose()
    # print the descriptives (\n inserts a line break)
    print('Balancing Checks:', '-' * 80,
          round(balance, 2), '-' * 80, '\n\n', sep='\n')
    # return results
    return balance
