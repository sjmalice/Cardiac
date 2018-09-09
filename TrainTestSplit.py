import pandas as pd
import numpy as np

import seaborn as sns

from Clean_Fun import *

# %%

cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
    'Cardiac_Meds','patient labs', 'patient BP', 'notes',"Old notes (Don't fit structure)"]
cpa=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name=cpa_sheets_load)
cpa.keys()

df=cpa['patient_enrollment_records']
df=df.drop(df.loc[df['patient_link'].apply(lambda x: True if len(str(x))<3 else False)].index)

df=outcome_split(df)

df2=df[df['train']==1]
test=df2[df2['outcome'].isnull()]
test[['patient_link','status','discharge','outcome','train','reason_for_dc']]
test.reason_for_dc.value_counts()

df.train.value_counts()
df.train.value_counts()
df[df.outcome.isnull()][['patient_link','status','discharge','outcome','train','reason_for_dc']]
cols=['patient_link','status','discharge','reason_for_dc','cardiac_related']#'Enrollment_Date',

df=df[cols]
df.head()
df_train=df[df['discharge']==True]
df_train.reason_for_dc.isnull().value_counts() # missing 136 out of 1100 times
df_train.reason_for_dc.value_counts()

outcome_dict={'Good':['To Home','No Reason Given','Assissted Living Facility','No Reason Given'], # CAN WE ASSUME THIS??? that In Nursing Facility
    'Bad':['Hospital','Death'],
    'Test':['In Nursing Facility','Skilled Nursing Facility (SNF)','Not approriate for program, removed']}

df[df['status'].isnull()]

# %%

sheets_to_load=['Hospitals', 'chains', 'facilities', 'Statuses', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

# Creates a dictionary with each sheet from the Excel, sheet_name=None means load all sheets
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=sheets_to_load)
cp.keys()
df2=cp['patient_enrollment_records'][cols]
df2.status.value_counts()


def outcome_split(df,outcome_dict={
    'Good':['To Home','No Reason Given','Assissted Living Facility','No Reason Given'], # CAN WE ASSUME THIS??? that In Nursing Facility
    'Bad':['Hospital','Death'],
    'Test':['In Nursing Facility','Skilled Nursing Facility (SNF)','Not approriate for program, removed']}):
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
outcome_split(df2).sample(10)
