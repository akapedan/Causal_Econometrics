"""
Data Analytics II: PC4 Functions.

Spring Semester 2022.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats


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


# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings and unique values
def my_adjstats(data):
    """
    Summary Statistics

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
        
    Returns
    -------
    None. Prints Summary Statistics Data Frame
    """
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
    # print the descriptives
    print('Descriptive Statistics:', '-' * 80,
          round(adjstats, 2), '-' * 80, '\n\n', sep='\n')


# own procedure to do histograms
def my_hist(data, varname, path, nbins=10, label=""):
    """
    Plot histograms.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    varname : TYPE: string
        DESCRIPTION: variable name for which histogram should be plotted
    path : TYPE: string
        DESCRIPTION: path where the plot will be saved
    nbins : TYPE: integer
        DESCRIPTION. Number of bins. The default is 10.
    label: Type: string
        DESCRIPTION. Label for the title. The default is none.

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    # add title
    if label == "":
        plot.title('Histogram of ' + varname)
    else:
        plot.title('Histogram of ' + varname + ' for ' + label)
    # add labels
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    if label == "":
        plot.savefig(path + '/histogram_of_' + varname + '.png')
    else:
        plot.savefig(path + '/histogram_of_' + varname + '_' + label + '.png')
    # print the plot
    plot.show()
    
# ATE estimation by mean differences
def ate_md(outcome, treatment, display=False):
    """
    Estimate ATE by differences in means.

    Parameters
    ----------
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treatment : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    display: TYPE: boolean
        DESCRIPTION: should results be printed?
        The default is False.

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
    ate_se = np.sqrt((y_1.var() / len(y_1)) + (y_0.var() / len(y_0)))
    ate_tval = ate / ate_se
    # compute pvalues based on the normal distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    ate_pval = stats.norm.sf(abs(ate_tval)) * 2  # twosided
    # alternatively ttest_ind() could be used directly
    # stats.ttest_ind(a=y_1, b=y_0, equal_var=False)
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    result = pd.DataFrame([ate, ate_se, ate_tval, ate_pval],
                          index=['ATE', 'SE', 'tValue', 'pValue'],
                          columns=['MeanDiff']).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, return and print result (\n inserts a line break)
        print('ATE Estimate by Difference in Means:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result


# own function for dummies check
def dummy_check(data, dummies, name):
    """
    Conduct a dummy variables check.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including dummies of interest
    dummies : TYPE: tuple of strings
        DESCRIPTION: names of dummies of interest
    name : TYPE: tuple of a string
        DESCRIPTION: name of dummies of interest

    Returns
    -------
    None. Prints dummy check table.
    """
    # create empty df
    dummy_table = pd.DataFrame(index=dummies + ('sum', ), columns=name)
    # identify the state
    state_id = 0 if name == ('Pennsylvania', ) else 1
    # fill in the table
    for dummy in dummies:
        dummy_table.loc[dummy, :] = np.mean(
            data.loc[data['state'] == state_id, dummy])
    # fill in the sum
    dummy_table.loc['sum', :] = sum(
        dummy_table.iloc[range(len(dummies)), :].values)
    # boolean check
    dummy_table.loc['check',:] = dummy_table.loc['sum',] == 1
    # print the table
    print('Dummy Check:', '-' * 80,
          dummy_table, '-' * 80, '\n\n', sep='\n')
    
# define own function for mean table
def mean_table(data, meanvar, byvar):
    """
    Mean table.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: Dataset including the variables of interest.
    meanvar : TYPE: tuple of strings
        DESCRIPTION: variable names for which means should be computed.
    byvar : TYPE: tuple of strings
        DESCRIPTION: variable names by which the means should ne computed.

    Returns
    -------
    None. Prints a mean table of variables of interest
    """
    # get unique values
    first_unique = list(data[byvar[0]].unique())
    second_unique = list(data[byvar[1]].unique())
    # create empty df
    meantable = pd.DataFrame(index=range(len(first_unique) +
                                         len(second_unique)),
                             columns=byvar + meanvar + ("N", ))
    # keep track of iterations for indexing
    idx = 0
    # for each unique values of the byvar compute the mean of meanvar
    for first_value in first_unique:
        for second_value in second_unique:
            # fill in values for state, year and observations
            meantable.loc[idx, byvar[0]] = ('Pennsylvania' if first_value == 0 else 'New Jersey')
            meantable.loc[idx, byvar[1]] = second_value
            meantable.loc[idx, 'N'] = len(data.loc[
                (data[byvar[0]] == first_value) &
                (data[byvar[1]] == second_value), :])
            # compute means for all meanvars
            for varname in meanvar:
                # get the means for subgroups
                meantable.loc[idx, varname] = np.mean(data.loc[
                    (data[byvar[0]] == first_value) &
                    (data[byvar[1]] == second_value), varname])
                # get iteration for indexing
            idx = idx + 1
    # print the mean table
    print('Mean Table:', '-' * 80,
          meantable, '-' * 80, '\n\n', sep='\n')