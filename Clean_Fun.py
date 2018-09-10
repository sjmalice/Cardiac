# Helper Cleaning Functions

import pandas as pd
import numpy as np
import re

def outcome_split(df,outcome_dict={
    'Good':['To Home','No Reason Given','Assissted Living Facility','No Reason Given'], # CAN WE ASSUME THIS??? that In Nursing Facility
    'Bad':['Hospital','Death'],
    'Test':['In Nursing Facility','Skilled Nursing Facility (SNF)',
    'Not approriate for program, removed']}):
    """ Input dataframe and Outcome dictionary
    Adds Train and Outcome columns to dataframe
    TO DO: Drop patient altogether if cardiac_related
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

def ef_deep_clean(x):
    """ helper function to clean_EF_rows
    extracts any digits from string
    recursively calls itself if there are too many digits
    """
    # remove EF from a previous record
    if re.search('previous',x):
        ind,__=re.search('previous',x).span()
        return ef_deep_clean(x[:ind])
    if re.search('(/)',x):
        ind,__=re.search('(/)',x).span()
        return ef_deep_clean(x[:ind-1])
    else: # Creates a list of digits
        tmp_dig=re.findall('\\b\\d+\\b', x)
        if len(tmp_dig)>2:
            print(x)
            return clean_EF_rows('pending')
        if len(tmp_dig)==2:
            return (float(tmp_dig[0])+float(tmp_dig[1]))/200.0
        if len(tmp_dig)==1:
            return clean_EF_rows(tmp_dig[0])
        # if there are really no digits, return na_val which corresponds to 'pending'
        if len(tmp_dig)==0:
            return clean_EF_rows('pending')

def clean_EF_rows(x,na_val=0.49,norm_val=0.55,list_strings=['pending','ordered','done','no data','new admission']):
    """ For use with a .apply(lambda) to the EF column
    ie. df['ef'].apply(lambda x: clean_EF_rows(x))
    Does not change NaN values, only messy string/percentages
    """
    #best case scenario: already a decimal or percentage with no sign
    x=str(x).replace('<','')
    x=str(x).replace('>','')
    try:
        if float(x)<1:
            return float(x)
        elif float(x)>10:
            return float(x)/100
    except:
        # For the percentages like 55%:
        x=str(x).replace('%','')
        # for percentage ranges like 50-55%
        try:
            st,en=re.search('-',x).span()
            # take the average
            return (float(x[:st])+float(x[en:]))/200.0
        except:
            if x.lower() in list_strings:
                return na_val
            elif re.search('normal',x.lower()):
                return norm_val
            else: # deep clean extracts digits from string text
                return ef_deep_clean(x)

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
            except:
                continue
        new_df=pd.concat([new_df, tmp_df], axis=0)
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
