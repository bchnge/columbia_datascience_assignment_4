import statsmodels.iolib.foreign as smio
import os
from pandas import DataFrame


""" This python script will convert raw stata files in data/raw to csvs in data/processed
"""

# set working path
current_dir = os.getcwd()
parent = os.path.join(current_dir, "..")
normal_parent = os.path.normpath(parent)

raw = normal_parent + '/data/raw/'
proc = normal_parent + '/data/processed/'

# Convert 2006
arr_06 = smio.genfromdta(raw+'2006.dta')
frame_06 = DataFrame.from_records(arr_06)
frame_06.to_csv(proc+'2006.csv', index = False)


# Convert 2010
arr_10 = smio.genfromdta(raw+'2010.dta')
frame_10 = DataFrame.from_records(arr_10)
frame_10.to_csv(proc+'2010.csv', index = False)

