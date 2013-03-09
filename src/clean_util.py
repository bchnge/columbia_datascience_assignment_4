# A set of helper utilies for cleaning Pandas series
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def elimMissVar(dframe, pct_required=0.5):
	""" takes in a dataframe and eliminates columns based on the percent of nonmissing required"""
	newframe = DataFrame(index = dframe.index)
	maxObs = dframe.shape[0]
	reqObs = maxObs * pct_required
	for col in dframe.columns:
		colct = dframe[col].count()
		if colct>=reqObs:
			newframe[col] = dframe[col]
			

def convertMissing(s, missingvalue):
    """ convertMissing will take a Series and replace all coded missing values (e.g. -999), with NaN"""
    A = Series(np.array(s).astype(float64))
    B = Series(A.replace(missingvalue, NaN))
    B.name = s.name
    return B


def disc2cont(s, endptlist):
	""" Transforms a discrete-valued series into a continuous-valued series based on list of endpoints using simple average"""
	vals = np.array(endptlist)
	newseries = s	
	for i in range(s):
		newseries = s.replace(i, 0.5*(vals[i] + vals[i-1]))
	return newseries


def get_r2(actual, estimate):
    
    """ read in actual and estimates (which are DataFrames) and output R2"""
    
    actual.columns = ['actual']
    estimate.columns = ['estimate']
    test = actual.join(estimate)
    
    # get SSE
    error = test['actual'] - test['estimate']
    errors = Series(error)
    errors = errors.dropna()
    SSE = errors.dot(errors)
    print 'SSE: ' + str(SSE)
    
    # get SST
    st = test['actual'][Series.notnull(test['estimate'])]
    avg_actual = mean(st, axis=1)
    st = st - avg_actual
    st_na = st.dropna()
    SST = st_na.dot(st_na)
    print 'SST: ' + str(SST)
    
    # calculate R2
    R2 = 1 - SSE/SST
    print 'R2: ' + str(R2)
    
    return R2


def cat2dummy(s):
	""" take a categorical series and return a dataframe of dummy variables for each value """
	dummyframe = DataFrame(index = s.index)	
	for elem in s.unique():
		dummyframe[str(elem)] = s == elem
	return dummyframe


