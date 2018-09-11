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

def clean_diastolic_columns(di_sys,bp,col_type):
    """ Imputes diastolic or systolic from the BP columns
    col_type distinguishes between di or sys
    Use like: df.apply(lambda row: clean_diastolic_columns(row['Diastolic'],row['resting_BP'],col_type='di'),axis=1)
    """
    try:
        if np.isnan(di_sys):
            sys_tmp,di_tmp=re.findall('\\b\\d+\\b', bp)
            if col_type=='di':
                return di_tmp
            elif col_type=='sys':
                return sys_tmp
            else:
                print("Error: please correct input variable col_type to be either 'di' or 'sys'")
        else:
            return di_sys
    except:
        pass

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

def impute_from_special_status(status_row,special_row):
    """ If status is empy and special status is Death, put Death into status
    use like:  df.apply(lambda row: impute_from_special_status(row['status'],row['special_status']),axis=1)
    """
    try:
        if np.isnan(status_row):
            if special_row=='Death':
                return 'Death'
            else:
                return status_row
    except:
        return status_row

def remove_invalid_rows(df):
    """ Takes the dataframe and removes specific instances where we have found
    invalid rows - when there is a row like: 1 2 3 ....
    or a test patient created by multitechvisions
    Check patient name for TEST, or for John Doe and Sally Test
    Should drop row "create_user", afterwards
    """
    # find the index of invalid rows
    ind_inv=df.loc[df['patient_link'].apply(lambda x: True if len(str(x))<3 else False)].index
    ind_inv=ind_inv.append(df.loc[df['patient_name'].apply(lambda x: search_for_test(x,'test'))].index)
    ind_inv=ind_inv.append(df.loc[df['patient_name'].apply(lambda x: search_for_test(x,'john doe'))].index)
    if len(ind_inv)!=0:
        # print and remove them
        for i in ind_inv:
            print("removing invalid row: "+str(i)+"\n")
            try:
                print(df.iloc[i][['patient_link','Enrollment_Date','patient_name','create_user']])
            except:
                print(df.iloc[i]['patient_link'])
            print('-'*50)
        # now remove them
        df.drop(ind_inv,axis=0,inplace=True)
    # reset the index before moving on
    df=df.reset_index()

    # Could remove this section since now we caught them with the 'test' search. But just in case
    # find observations from Test
    ind_test=df.loc[df['create_user']=='multitechvisions@gmail.com'].index#[['patient_link','Enrollment_Date','patient_name','create_user']]
    if len(ind_test)!=0:
        df.iloc[ind_test]
        print("removing multitechvisions test rows: \n")
        try:
            print(df.iloc[ind_test][['patient_link','Enrollment_Date','patient_name','create_user']])
        except:
            print(df.iloc[ind_test]['patient_link'])
        print('-'*50)
        df.drop(ind_test,axis=0,inplace=True)
    print('\n \n Dropped '+str(len(ind_inv)+len(ind_test))+' rows from the dataset')
    print('New size of dataset: '+str(df.shape))
    return df

# maybe there's an errors coerce function
def search_for_test(x,search_word):
    """ Handles errors and search for 'test' in the input variable
    Use as df.loc[df['patient_name'].apply(lambda x: search_for_test(x,'test'))]
    """
    try:
        return re.search(search_word,x.lower())!= None
    except:
        return False
