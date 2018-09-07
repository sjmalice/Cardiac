# Helper Cleaning Functions

import pandas as pd
import numpy as np

def choose_most_recent(df,date_col):
    ''' Choose the most recent lab/test result from list of results
    To Do: make the drop duplicates more robust since misses some patients
    '''
    new_df=pd.DataFrame(columns=df.columns)
    for pat in df.patient_link.unique():
        pat_df=df.loc[df.patient_link==pat]
        rows,col =pat_df.shape
        if rows==1:
            tmp_df=pat_df
        else:
            try:
                tmp_df=pat_df.loc[pat_df[date_col]==max(pat_df[date_col])]
                tmp_df.reset_index(inplace=True)
            except:
                continue
        new_df=pd.concat([new_df, tmp_df.iloc[[0]]], axis=0)
    return new_df.drop_duplicates()
