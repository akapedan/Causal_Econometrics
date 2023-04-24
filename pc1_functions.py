"""
Data Analytics II: PC1 Functions.

Spring Semester 2022.

University of St. Gallen.
"""

# import modules here
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# set working directory
PATH = '/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/PC/PC1'
sys.path.append(PATH)

# Exercise 1b) 
# Function including mean, variance, standard deviation, maximum and minimum, 
# the number of missing and unique values and number of observations 
# as well as the variable names in a single object
def my_adjstats(data):
    # create empty df with fitting index
    columns = ['Mean','Variance', 'Std', 'Min', 'Max', 'NA', 'Unique values', 'N']
    index = data.columns
    adjstats = pd.DataFrame(index=index, columns=columns)
    # fill empty df with respective values
    adjstats['Mean'] = data.mean()
    adjstats['Variance'] = np.var(data)
    adjstats['Std'] = np.std(data)
    adjstats['Max'] = data.max()
    adjstats['Min'] = data.min()
    adjstats['NA'] = np.count_nonzero(np.isnan(data))
    adjstats['Unique values'] = data.nunique()
    adjstats['N'] = data.count()
    # show df
    print(adjstats)

# Exercise 1e)
def my_hist(data, save):
    # create new df with only continuous i.e. float variables
    data_float = data.select_dtypes('float')
    # define variables as column names
    variables=data_float.columns
    # create loop that creates a histogram for each variable
    for i in variables:
        plot=plt.hist(data[i], bins=30)
        plt.title(i)
        # argument save if user wants to save the 
        # histograms in the corresponding directory
        if save == True:
            plt.savefig(PATH + '/graph_{}.png'.format(i))
        plt.show()

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
        sdiff = abs(mdiff / np.sqrt((treated.var() + control.var()) / 2)) * 100
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


# ATE estimation by mean differences
def ate_md(outcome, treatment):
    """
    Estimate ATE by differences in means.

    Parameters
    ----------
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treatment : TYPE: pd.Series
        DESCRIPTION: vector of treatments

    Returns
    -------
    results : ATE with Standard Error
    """
    # outcomes y according to treatment status by logical vector of True/False
    # set treated and control apart using the location for subsetting
    # using the .loc both labels as well as booleans are allowed
    y_1 = outcome.loc[treatment == 1]
    y_0 = outcome.loc[treatment == 0]
    # compute ATE and its standard error and t-value
    ate = y_1.mean() - y_0.mean()
    ate_se = np.sqrt(y_1.var() / len(y_1) + y_0.var() / len(y_0))
    ate_tval = ate / ate_se
    # get the degrees of freedom (unequal variances, Welch t-test)
    d_f = (ate_se ** 4 /
           (((y_1.var() ** 2) / ((len(y_1) ** 2) * (len(y_1) - 1))) +
            ((y_0.var() ** 2) / ((len(y_0) ** 2) * (len(y_0) - 1)))))
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    ate_pval = stats.t.sf(np.abs(ate_tval), d_f) * 2
    # alternatively ttest_ind() could be used directly
    # stats.ttest_ind(a=y_1, b=y_0, equal_var=False)
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    result = pd.DataFrame([ate, ate_se, ate_tval, ate_pval],
                          index=['ATE', 'SE', 'tValue', 'pValue'],
                          columns=['MeanDiff']).transpose()
    # return and print result (\n inserts a line break)
    print('ATE Estimate by Difference in Means:', '-' * 80,
          'Dependent Variable: ' + outcome.name, '-' * 80,
          round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result


# own procedure for testing the differences
def test_diff(means, serrors, nobs):
    """
    Test for differences in means.

    Parameters
    ----------
    means : TYPE: tuple
        DESCRIPTION: means of sample 1 and sample 2
    serrors : TYPE: tuple
        DESCRIPTION: standard errors of sample 1 and sample 2
    nobs : TYPE: tuple
        DESCRIPTION: number of observations in sample 1 and sample 2

    Returns
    -------
    result: mean, se, t-value and p-value for test if difference is zero.
    """
    # get the difference of the means
    mean_diff = means[0] - means[1]
    # get standard error for the difference
    se_diff = np.sqrt(serrors[0] ** 2 + serrors[1] ** 2)
    # get t-value
    t_val = mean_diff / se_diff
    # get the degrees of freedom (unequal variances, Welch test)
    d_f = (se_diff ** 4 / (((serrors[0] ** 4) / (nobs[0] - 1)) +
                           ((serrors[1] ** 4) / (nobs[1] - 1))))
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    p_val = stats.t.sf(np.abs(t_val), d_f) * 2
    # bind result into dataframe
    result = pd.DataFrame([mean_diff, se_diff, t_val, p_val],
                          index=['coef', 'se', "t-value", "p-value"],
                          columns=['Diff in Means']).transpose()
    # print the result (\n inserts a line break)
    print('Results for Test if Difference in Means is Zero:', '-' * 80,
          round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the result
    return result


