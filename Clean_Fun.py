# Helper Cleaning Functions

import pandas as pd
import numpy as np
import gspread
from oauth2client import file, client, tools

def outcome_split(df,outcome_dict={
    'Good':['To Home','No Reason Given','Assissted Living Facility','No Reason Given'], # CAN WE ASSUME THIS??? that In Nursing Facility
    'Bad':['Hospital','Death'],
    'Test':['In Nursing Facility','Skilled Nursing Facility (SNF)',
    'Not approriate for program, removed']}):
    """ Input dataframe and Outcome dictionary
    Adds Train and Outcome columns to dataframe
    """
    outcome={}
    train={}
    for row in range(df.shape[0]):
        if df.iloc[row]['status'] in outcome_dict['Good']:
            outcome[df.iloc[row]['patient_link']]=1
            train[df.iloc[row]['patient_link']]=1
        if df.iloc[row]['status'] in outcome_dict['Bad']:
            outcome[df.iloc[row]['patient_link']]=0
            train[df.iloc[row]['patient_link']]=1
        if df.iloc[row]['status'] in outcome_dict['Test']:
            train[df.iloc[row]['patient_link']]=0
        elif df.iloc[row]['discharge']==True:
            train[df.iloc[row]['patient_link']]=1
        elif df.iloc[row]['discharge']==False:
            train[df.iloc[row]['patient_link']]=0
    df['outcome']=df['patient_link'].map(outcome)
    df['train']=df['patient_link'].map(train)
    return df

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

def lower_errors(x):
    try:
        return x.lower()
    except:
        return ""

def find_unique_diag(df_diag_column):
    """
    Within text Diagnosis Columns, returns a list of the Unique Diagnoses,
    removing the combinations of diagnoses
    """
    all_diag=df_diag_column.apply(lambda x: lower_errors(x)).unique()
    all_diag[7].split(' , ')
    unique_diag=[]
    for diag in all_diag:
        if len(diag)==0:
            continue
        else:
            unique_diag.append(diag.split(' , '))
    flat_list = [item for sublist in unique_diag for item in sublist]
    unique_diag=pd.Series(flat_list).unique()
    return unique_diag

def dummify_diagnoses(df,unique_diag,diagnosis_col='Diagnosis_1'):
    """
    Takes Diagnoses and dummifies them for patients. If a patient has multiple
    diagnoses, will put a 1 in all relevant Diagnoses.
    The kth column is NA, no diagnosis. Maybe we will impute with the mode?
    """
    header=unique_diag.tolist().append('patient_link')
    dummy_diag=pd.DataFrame(columns=header)

    for row in range(df.shape[0]):
        pat_diag=lower_errors(df.iloc[row][diagnosis_col]).split(' , ')
        # print(pat_diag)
        dict_dummy_diag=dict(zip(unique_diag,np.zeros(len(unique_diag))))
        # dict_dummy_diag['patient_link']=df.iloc[row]['patient_link']
        #pd.DataFrame(np.zeros(len(unique_diag)).reshape(-1),columns=unique_diag)
        for diag in pat_diag:
            if diag in unique_diag:
                dict_dummy_diag[diag]=1
            else:
                continue
        tmp_dummy_diag=pd.DataFrame(dict_dummy_diag, index=[row])
        tmp_dummy_diag['patient_link']=df.iloc[row]['patient_link']
        dummy_diag = pd.concat([dummy_diag,tmp_dummy_diag], axis=0)

    return dummy_diag
