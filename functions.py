"""
Data Analytics II: Simulation Study Functions.

Author: Arbian Halilaj, 16-609-828.

Spring Semester 2022.

University of St. Gallen.
"""
# load the required functions
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

##############################################################################

def dgp1(dim_x,n):
    '''Creates one draw of DGP1

    Parameters
    ----------
    dim_x : Number of Covariates
    n : ONumber of Observations
    
    Returns
    -------
    x : Covariates
    d : Treatment
    y : Outcome
    
    '''
    x = np.random.normal(0, 1, size=(n, dim_x))
    betas = np.array(list(range(1,dim_x+1)))/(dim_x)
    g = x @ betas
    d = np.random.binomial(1,0.5,n)
    y0 = np.random.normal(0,1,n)
    y1 = -1 + g + np.random.normal(0,1,n) 
    y = d * y1 + (1-d) * y0
    return(x,d,y)

def dgp2(dim_x,n):
    '''Creates one draw of DGP2

    Parameters
    ----------
    dim_x : Number of Covariates
    n : ONumber of Observations
    
    Returns
    -------
    x : Covariates
    d : Treatment
    y : Outcome

    '''
    x = np.random.normal(0, 1, size=(n, dim_x))
    betas = np.array(list(range(1,dim_x+1)))/(dim_x)
    d = np.random.binomial(1,0.8,n)
    g = x @ betas 
    y0 = np.random.normal(0,1,n)
    y1 = -1 + g + np.random.normal(0,1,n) 
    y = d * y1 + (1-d) * y0
    return(x,d,y)

def dgp3(dim_x,n):
    '''Creates one draw of DGP3

    Parameters
    ----------
    dim_x : Number of Covariates
    n : ONumber of Observations
    
    Returns
    -------
    x : Covariates
    d : Treatment
    y : Outcome

    '''
    x = np.random.normal(0, 1, size=(n, dim_x))
    betas = np.array(list(range(1,dim_x+1)))/(dim_x)
    g = x @ betas
    v = np.random.uniform(0,1,n)
    d = np.random.binomial(1,0.5*v,n)
    y1 = 1 + g + np.random.normal(0,1,n)
    y0 = np.random.normal(0,1,n) 
    y = d * y1 + (1-d) * y0 + v
    return(x,d,y)

##############################################################################

def ols(x,y):
    '''Estimates the ATE according to Doubly OLS

    Parameters
    ----------
    x : Covariates
    y : Output
    
    Returns
    -------
    Returns the ATE.

    '''
    n = y.shape[0]          # num of obs
    x_c = np.c_[np.ones(n),x] # add constant
    betas = np.linalg.inv(x_c.T @ x_c) @ x_c.T @ y # calculate coeff
    return(betas)

def dr(d, x, y, dim_x):
    '''Estimates the ATE according to Doubly Robust approach

    Parameters
    ----------
    d : Treatment variable
    x : Covariates
    y : Output
    dim_x : Number of Covariates
    
    Returns
    -------
    Returns the ATE.

    '''
    # Step 1: Estimate propensity score with logit
    ps = sm.Logit(endog = d, exog = sm.add_constant(x)).fit(disp = 0).predict()
    
    # Step 2: Estimate outcome equation by ols
    data = np.column_stack([d, x, y])
    data = pd.DataFrame(data)
    
    data.columns=["V"+str(i) for i in range(1, dim_x + 3)] # give arbitrary column names
    
    df1 = data.query("V1==1") 
    df0 = data.query("V1==0")
    
    x1 = df1.iloc[:,1:dim_x+1]
    x0 = df0.iloc[:,1:dim_x+1]
    
    y1 = df1.iloc[:,dim_x+1]
    y0 = df0.iloc[:,dim_x+1]
    
    mu1 = LinearRegression().fit(x1, y1).predict(x1)
    mu0 = LinearRegression().fit(x0, y0).predict(x0)

    # Step 3: Calculate ATE
    ate = np.mean(d*(y - np.c_[mu1])/ps + np.c_[mu1]) - np.mean((1-d)*(y - np.c_[mu0])/(1-ps) + np.c_[mu0])
    
    # Create a vector of the value
    ate = np.c_[ate]
    
    # Return the result
    return ate

##############################################################################

def simulation(n_sim, n, dim_x):
    '''Runs a simulation over all GDP's

    Parameters
    ----------
    n_sim : Number of Simulations
    n : Number of Obervations
    dim_x : Number of Covariates
    
    Returns
    -------
    Returns the ATE.

    '''
    all_results = np.empty( (n_sim, 2, 3) )  # initialize for results

    # Loop through many simulations
    for i in range(n_sim):
        # Run DGP1
        x, d, y = dgp1(dim_x,n)
        all_results[i,0,0] = dr(d, x, y, dim_x)
        all_results[i,1,0] = ols(np.c_[d,x],y)[1]

        # Run DGP2
        x, d, y = dgp2(dim_x,n)
        all_results[i,0,1] = dr(d, x, y, dim_x)
        all_results[i,1,1] = ols(np.c_[d,x],y)[1]
        
        # Run DGP3
        x, d, y = dgp3(dim_x,n)
        all_results[i,0,2] = dr(d, x, y, dim_x)
        all_results[i,1,2] = ols(np.c_[d,x],y)[1]
        
    return all_results

def plot_results(results,dgp,truth):
    plt.figure()
    plt.hist(x=results[:,1,dgp-1], bins='auto', color='red',alpha=0.5,label="OLS")
    plt.hist(x=results[:,0,dgp-1], bins='auto', color='blue',alpha=0.5,label="Doubly Robust")
    plt.axvline(x=truth,label="truth")
    plt.legend(loc='upper right')
    plt.title('DGP' + str(dgp))
    plt.show()

    
def print_results(results,dgp,truth):
    # Caculate and print performance measures
    bias_ols = np.mean(results[:,1,dgp-1]) - truth
    variance_ols= np.mean( (results[:,1,dgp-1] - np.mean(results[:,1,dgp-1]))**2 )
    mse_ols= bias_ols**2 + variance_ols
    bias_dr = np.mean(results[:,0,dgp-1]) - truth
    variance_dr = np.mean( (results[:,0,dgp-1] - np.mean(results[:,0,dgp-1]))**2 )
    mse_dr = bias_dr**2 + variance_dr
    print("\n Results DGP" + str( dgp ))
    print("Bias OLS: " + str( round(bias_ols,3) ))
    print("Bias DR: " + str( round(bias_dr,3) ))
    print("Variance OLS: " + str( round(variance_ols,3) ))
    print("Variance DR: " + str( round(variance_dr,3) ))
    print("MSE OLS: " + str( round(mse_ols,3) ))
    print("MSE DR: " + str( round(mse_dr,3) ))

##############################################################################
# APPENDIX: Doubly ML Algorithm 
##############################################################################
'''
def dml(d, x, y, numberOfFolds, n_x):
    # Get data from DGP and store into DataFrame
    data = np.column_stack([d, x, y])
    data = pd.DataFrame(data)
        
    # 1) Split data
    dataList = np.array_split(data.sample(frac=1), numberOfFolds)
    result = []
    
    # Get nuisance estimator
    nuisanceEstimatorG = RandomForestRegressor(max_depth=30, max_features='sqrt', n_estimators=500, min_samples_leaf=2)
    nuisanceEstimatorM = RandomForestRegressor(max_depth=30, max_features='sqrt', n_estimators=500, min_samples_leaf=2)
    
    for i in range(len(dataList)):
            
        # Prepare D (treatment effect), Y (predictor variable), X (controls)
        mainData = dataList[0]
        D_main = mainData.iloc[:,0].values
        Y_main = pd.DataFrame(mainData.iloc[:,n_x+1])
        X_main = pd.DataFrame(mainData.iloc[:,1:n_x+1])
            
        dataList_ = dataList[:]
        dataList_.pop(0)
        compData = pd.concat(dataList_)
        D_comp = compData.iloc[:,0].values
        Y_comp = pd.DataFrame(compData.iloc[:,n_x+1])
        X_comp = pd.DataFrame(compData.iloc[:,1:n_x+1])
            
        # Compute g as a machine learning estimator, which is trained on the predictor variable 
        g_comp = nuisanceEstimatorG.fit(X_main, Y_main.values.ravel()).predict(X_comp)
        g_main = nuisanceEstimatorG.fit(X_comp, Y_comp.values.ravel()).predict(X_main)
            
        # Compute m as a machine learning estimator, which is trained on the treatment variable
        m_comp = nuisanceEstimatorM.fit(X_main, D_main).predict(X_comp)
        m_main = nuisanceEstimatorM.fit(X_comp, D_comp).predict(X_main)
            
        # Compute V
        V_comp = np.array(D_comp) - m_comp
        V_main = np.array(D_main) - m_main
            
        # We provide two different theta estimators for computing theta
        theta_comp = pc.thetaEstimator(Y_comp, V_comp, D_comp, g_comp)
        theta_main = pc.thetaEstimator(Y_main, V_main, D_main, g_main)
        result.append((np.mean(theta_comp + theta_main)))

    # Aggregate theta
    theta = np.mean(result)

    return theta

def thetaEstimator(Y, V, D, g):
    try:
        return np.mean((np.array(V)*(np.array(Y)-np.array(g))))/np.mean((np.array(V)*np.array(D)))
    except ZeroDivisionError:
        return 0
'''