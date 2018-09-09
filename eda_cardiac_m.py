import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from Clean_Fun import *

''' All sheets from CP:
 ['Menu', 'Notifications', 'users', 'Hospitals', 'Reports',
'Bundle_Reports', 'chains', 'facilities', 'Statuses', 'BGs', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

Some notes on them:
Sheet30 - Just building the visual for how many days they've been in hospital

To do:
Notes sheet - determine patient_link
'''

sheets_to_load=['Hospitals', 'chains', 'facilities', 'Statuses', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

# Creates a dictionary with each sheet from the Excel, sheet_name=None means load all sheets
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=sheets_to_load)
cp.keys()
cp['patient_enrollment_records'].columns

# this helped me realize that records from the archive, have their Enrollment Records in the not-Archive. So merging is a necessity
# cp['patients'].loc[cp['patients']['patient_id']=='W6X7LdR5']
# cp['patient_enrollment_records'].loc[cp['patient_enrollment_records']['patient_link']=='W6X7LdR5']

# %% first explore Patients tab

df=cp['patients']

df.columns

cut_columns =['Date_of_Birth', 'Patient_Gender', 'EF', 'Date of last echo',
        'cardiac_plan', 'patient_id','edit_time_stamp','Current_Facility_value', 'Current_Chain_value',
       'Report', 'Hospital_History', 'Archive_Patient', 'Last_Weight',
       'Weight_Change', 'Weight_Change_Since_Admit', 'Last_BNP', 'BNP_Change',
       'BNP', 'Last_Labs', 'CR_Change', 'Last_Meds', 'Admin Notes',
       'special_status']
df=df[cut_columns]
df.columns=['Date_of_Birth', 'Patient_Gender', 'EF', 'Date of last echo',
        'cardiac_plan', 'patient_link','edit_time_stamp','Current_Facility_value', 'Current_Chain_value',
       'Report', 'Hospital_History', 'Archive_Patient', 'Last_Weight',
       'Weight_Change', 'Weight_Change_Since_Admit', 'Last_BNP', 'BNP_Change',
       'BNP', 'Last_Labs', 'CR_Change', 'Last_Meds', 'Admin Notes',
       'special_status']
len(df.patient_link.unique())

# check for bad rows to drop
df.loc[df['patient_link'].apply(lambda x: True if len(str(x))<3 else False)]

# %% Enrollment Records

per=cp['patient_enrollment_records']

cut_columns2=['patient_link', 'Enrollment_Date', 'Hospital_discharged_from',
       'Hospital_Admit_Date', 'Admit_weight', 'Diagnosis_1', 'Diagnosis_2',
       'Acute_or_chronic', 'AICD', 'status', 'discharge',
       'to_which_hospital', 'reason_for_dc', 'discharge_date',
       'cardiac_related', 'Enrollment_Active', 'Chain_link']
per=per[cut_columns2]
per['discharge'].value_counts()
per['discharge'].isnull().sum()
per_train=per[per['discharge']==True]
train_patients=per[per['discharge']==True]['patient_link']
len(train_patients.unique())

enroll_date=choose_most_recent(per_train,'Enrollment_Date')

enroll_date.shape

df=df.loc[df['patient_link'].apply(lambda x: True if (x in train_patients.values) else False)]
df=choose_most_recent(df,'edit_time_stamp')
df.shape

df_total=pd.merge(df,enroll_date,on='patient_link',how='outer')
df_total.shape

df_total.to_csv('CleanData/CardiacM_clean.csv')
# %%

plt.subplots(figsize=(20,15))
heat=sns.heatmap(df_total.isnull(), cbar=False)
# plt.xticks(rotation=75)
fig=heat.get_figure()
# fig.savefig('cardiac_m_missingness.png',transparent=True, dpi=400,bbox_inches='tight' ,format='png')

# %%
