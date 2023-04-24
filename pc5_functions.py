"""
Data Analytics II: PC5 Functions.

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
def my_hist(data, varname, path, nbins=10):
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

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    # add labels
    plot.title('Histogram of ' + varname)
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    plot.savefig(path + '/histogram_of_' + varname + '.png')
    # print the plot
    plot.show()
    
def employed_kidcount(data, varname, groupname):
    """
    Check mean and number of observations for each value of kidcount.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing variables of interest.
    varname : TYPE: string
        DESCRIPTION: variable name for which mean should be calculated
    groupname : TYPE: string
        DESCRIPTION: variable name for which group the mean of varname should be calculated
        
    Returns
    -------
    None. Prints df.

    """
    # subset data
    df = data[[varname,groupname]]
    
    # use dictionary to group for kidcount
    df = dict(tuple(df.groupby(groupname)))

    # looping around a dictionary seems not to be feasible because of locals
    # hence we just assign the dfs manually
    df2 = df[2]
    df3 = df[3]
    df4 = df[4]
    df5 = df[5]
    df6 = df[6]
    df7 = df[7]
    df8 = df[8]
    df9 = df[9]
    df10 = df[10]
    df11 = df[11]
    
    # create empty dataframe to fill with data after
    columns = [groupname, varname + '_mean', 'sample size']
    employed_means = pd.DataFrame(columns=columns)
    # fill column "kidcount" with values from 2 to 11
    employed_means['kidcount'] = list(range(2, 12))
    # fill all means into the mean column
    employed_means[varname + '_mean'] = (df2['employed'].mean(),
                                       df3['employed'].mean(),
                                       df4['employed'].mean(),
                                       df5['employed'].mean(),
                                       df6['employed'].mean(),
                                       df7['employed'].mean(),
                                       df8['employed'].mean(),
                                       df9['employed'].mean(),
                                       df10['employed'].mean(),
                                       df11['employed'].mean())
    
    # fill all sample sizes into the sample size column
    employed_means['sample size'] = (len(df2),
                                     len(df3),
                                     len(df4),
                                     len(df5),
                                     len(df6),
                                     len(df7),
                                     len(df8),
                                     len(df9),
                                     len(df10),
                                     len(df11))
    # show results
    print('Mean of employed for each number of kids:', '-' * 80,
          round(employed_means, 2), '-' * 80, '\n\n', sep='\n')
    
def crosstable(data, var1, var2):
    """
    Show relationship between morekids and multi2nd in a cross table.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing variables of interest.
    var1 : TYPE: string
        DESCRIPTION: variable name 
    var2 : TYPE: string
        DESCRIPTION: variable name     

    Returns
    -------
    None. Prints crosstable.

    """
    # subset data
    df = data[[var1,var2]]
    
    # calculate cell values
    cross00 = df[(df[var1] == 0) & (df[var2] == 0)]
    cross01 = df[(df[var1] == 0) & (df[var2] == 1)]
    cross10 = df[(df[var1] == 1) & (df[var2] == 0)]
    cross11 = df[(df[var1] == 1) & (df[var2] == 1)]
    
    cross0 = [len(cross00), len(cross10)]
    cross1 = [len(cross01), len(cross11)]
    
    # create empty df
    rows = ['morekids=0', 'morekids=1']
    columns = ['multi2nd=0', 'multi2nd=1']
    
    cross_table = pd.DataFrame(columns=columns, index=rows)

    # fill df
    cross_table['multi2nd=0'] = cross0
    cross_table['multi2nd=1'] = cross1

    # show cross table
    print(cross_table)


# use own ols procedure
def my_ols(exog, outcome, instrument, endogname, intercept=True, display=True, SLS=True):
    """
    OLS estimation.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: outcome
    intercept : TYPE: boolean
        DESCRIPTION: should intercept be included? The default is True.
    display : TYPE: boolean
        DESCRIPTION: should results be displayed? The default is True.

    Returns
    -------
    None. Prints OLS estimation results.
    """
    # check if intercept should be included
    # the following condition checks implicitly if intercept == True
    if intercept:
        # if True, prepend a vector of ones to the covariate matrix
        exog = pd.concat([pd.Series(np.ones(len(exog)), index=exog.index,
                                    name='intercept'), exog], axis=1)
    # compute (x'x)-1 by using the linear algebra from numpy
    x_inv_ols = np.linalg.inv(np.dot(exog.T, exog))
    # estimate betas according to the OLS formula b=(x'x)-1(x'y)
    betas_ols = np.dot(x_inv_ols, np.dot(exog.T, outcome))
    
    # compute the residuals by subtracting fitted values from the outcomes
    res_ols = outcome - np.dot(exog, betas_ols)
    
    # estimate standard errors for the beta coefficients
    # se = square root of diag((u'u)(x'x)^(-1)/(N-p))
    s_e_ols = np.sqrt(np.diagonal(np.dot(np.dot(res_ols.T, res_ols), x_inv_ols) /
                              (exog.shape[0] - exog.shape[1])))
    # compute the t-values
    tval_ols = betas_ols / s_e_ols
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    pval_ols = stats.t.sf(np.abs(tval_ols),
                      (exog.shape[0] - exog.shape[1])) * 2
    # put results into dataframe and name the corresponding values
    result_ols = pd.DataFrame([betas_ols, s_e_ols, tval_ols, pval_ols],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=list(exog.columns.values)).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, print the results (\n inserts a line break)
        print('OLS Estimation Results:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result_ols, 2), '-' * 80, '\n\n', sep='\n')
    
    if SLS: 
    # save fitted values in sep df
        fitted = np.dot(exog, betas_ols)

    # remove endogenous variable from exog df
        exog_sls = exog.drop(endogname, 1)
    
    # add IV to exog
        instrument = pd.DataFrame(instrument)
        instrumentname = instrument.columns.values.tolist()
        exog_sls[instrumentname] = instrument
            
    # compute (x'x)-1 by using the linear algebra from numpy
        x_inv_sls = np.linalg.inv(np.dot(exog_sls.T, exog_sls))
        
    # estimate betas according to the OLS formula b=(x'x)-1(x'y)
        betas_sls = np.dot(x_inv_sls, np.dot(exog_sls.T, fitted))
    
    # compute the residuals by subtracting fitted values from the outcomes
        res_sls = fitted - np.dot(exog_sls, betas_sls)
    
    # estimate standard errors for the beta coefficients
        s_e_sls = np.sqrt(np.diagonal(np.dot(np.dot(res_sls.T, res_sls), x_inv_sls) /
                              (exog_sls.shape[0] - exog_sls.shape[1])))
    # compute the t-values
        tval_sls = betas_sls / s_e_sls
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
        pval_sls = stats.t.sf(np.abs(tval_sls),
                          (exog_sls.shape[0] - exog_sls.shape[1])) * 2
    # put results into dataframe and name the corresponding values
        result_sls = pd.DataFrame([betas_sls, s_e_sls, tval_sls, pval_sls],
                              index=['coef', 'se', 't-value', 'p-value'],
                              columns=list(exog_sls.columns.values)).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
        if display:
        # if True, print the results (\n inserts a line break)
            print('2SLS Estimation Results:', '-' * 80,
                  'Dependent Variable: ' + endogname, '-' * 80,
                  round(result_sls, 2), '-' * 80, '\n\n', sep='\n')
            
    
