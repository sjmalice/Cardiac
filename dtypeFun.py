import pandas as pd
import pickle

def columnFormater(df, pklPath):
    '''
    Sets the dtype of each column in the data frame based on given dictionary

    Keyword Arguments
    -----------------
    df -- The data frame whose columns are to have their dtypes set
    pklPath -- file path of pickle containing dtype dictionary as string

    WARNING -- This is a mutating function
    '''
    # read the column format pickle
    with open(pklPath, 'rb') as f:
        dtypeDict = pickle.load(f)
    #change date columns to datetime dtype
    df[dtypeDict['datetimes']]\
    = df[dtypeDict['datetimes']].apply(pd.to_datetime)
    print("="*40)
    print("{} columns set to datetime".format(dtypeDict['datetimes']))
    # Change numeric columns to numeric dtype
    df[dtypeDict["numeric"]] = df[dtypeDict["numeric"]].apply(pd.to_numeric)
    print("="*40)
    print("{} columns set to numeric".format(dtypeDict['numeric']))
    print("="*40)
