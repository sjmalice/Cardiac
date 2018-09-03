import pandas as pd
import numpy as np

''' All sheets from CP:
 ['Menu', 'Notifications', 'users', 'Hospitals', 'Reports',
'Bundle_Reports', 'chains', 'facilities', 'Statuses', 'BGs', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

Some notes on them:
Sheet30 - Just building the visual for how many days they've been in hospital

To do:
Notes sheet - determine patient_link

All sheets from CPA:
['patient_enrollment_records', 'Settings', 'patient weights', 'patient BNP',
'Cardiac_Meds', 'patient labs', 'patient BP', 'notes',
"Old notes (Don't fit structure)"]

To do:
If we have time, clean up Notes and Old Notes

 '''
sheets_to_load=['Hospitals', 'chains', 'facilities', 'Statuses', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

# Creates a dictionary with each sheet from the Excel, sheet_name=None means load all sheets
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=sheets_to_load)
cp['patient_enrollment_records'].columns
cp['patients'].loc[cp['patients']['patient_id']=='W6X7LdR5']['Patient_Name','','Admin Notes']
cp['patient_enrollment_records'].loc[cp['patient_enrollment_records']['patient_link']=='W6X7LdR5']
cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
'Cardiac_Meds','patient labs', 'patient BP', 'notes',"Old notes (Don't fit structure)"]
cpa=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name=cpa_sheets_load)
cpa.keys()

# Attempt to simply separate Patients that have been discharged or not:
# TO DO - remove row 1059 which is MESS
df=cpa['patient_enrollment_records']

# remove one odd row - will remove any row with no valid patient link
df=df.drop(df.loc[df['patient_link'].apply(lambda x: True if len(str(x))<3 else False)].index)

cpa['patient_enrollment_records'].dtypes
df['discharge'].value_counts()

#Training is where discharge == True
df_train=df[df['discharge']==True]
train_patients=df[df['discharge']==True]['patient_link']

keep_columns=['patient_link', 'Enrollment_Date',
       'Hospital_discharged_from',
       'Hospital_Admit_Date', 'Admit_weight', 'Diagnosis_1', 'Diagnosis_2',
       'Acute_or_chronic', 'AICD', 'status', 'discharge',
       'to_which_hospital', 'reason_for_dc', 'discharge_date',
       'cardiac_related', 'Chain_link']
df_train.loc[df_train['patient_link']=='W6X7LdR5']
# we could use df_train patient links to always separate the dataset
df_train=df_train[keep_columns]
weight=cpa['patient weights']
weight.shape
weight_columns=['patient_link', 'patient_weight_date', 'weight',
       'Next_weight_due', 'editcount', 'edit_time_stamp',
       'facility_Link', 'Chain_link', 'Hospitals', 'This_Weight_Change',
       'Weight_Change_Since_Admit']

weight=weight.loc[weight['patient_link'].apply(lambda x: True if (x in train_patients.values) else False)]
weight=weight[weight_columns]
weight.loc[weight.patient_link=='W6X7LdR5']
# we don't have the weight for every patient

import matplotlib.pyplot as plt
weight.patient_link.value_counts().plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
plt.title('Number of records of Patient weights')
plt.xlabel('Number of times 1 patient was weighed')
plt.ylabel('Frequency')


cpa.keys()
