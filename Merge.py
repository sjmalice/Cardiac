# Merge Cardiac and Archive

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

cardiac=pd.read_csv('CardiacM_clean.csv')
archive=pd.read_csv('ArchiveClean.csv')
archive.shape
cardiac.shape
# %%

cardiac=cardiac.drop('Unnamed: 0',axis=1)
archive=archive.drop('Unnamed: 0',axis=1)

test=pd.merge(cardiac,archive,on='patient_link',how='outer')
test.shape

# %% Testing for duplicates

test.loc[test.patient_link=='W03ifrQD']
test.patient_link.value_counts()#.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
len(test.patient_link.unique())

# %%

plt.subplots(figsize=(20,15))
heat=sns.heatmap(test.isnull(), cbar=False)
# plt.xticks(rotation=75)
fig=heat.get_figure()
fig.savefig('cardiac_merge.png',transparent=True, dpi=400,bbox_inches='tight' ,format='png')

# %% Figure out a better merge system

test.columns
#Cardiac
['Date_of_Birth', 'Patient_Gender', 'EF', 'Date of last echo',
       'cardiac_plan', 'patient_link', 'edit_time_stamp',
       'Current_Facility_value', 'Current_Chain_value', 'Report',
       'Hospital_History', 'Archive_Patient', 'Last_Weight', 'Weight_Change',
       'Weight_Change_Since_Admit', 'Last_BNP', 'BNP_Change', 'BNP',
       'Last_Labs', 'CR_Change', 'Last_Meds', 'Admin Notes', 'special_status',
       'Enrollment_Date', 'Hospital_discharged_from', 'Hospital_Admit_Date',
       'Admit_weight', 'Diagnosis_1', 'Diagnosis_2', 'Acute_or_chronic',
       'AICD', 'status', 'discharge', 'to_which_hospital', 'reason_for_dc',
       'discharge_date', 'cardiac_related', 'Enrollment_Active', 'Chain_link']
#Archive
['patient_link', 'Enrollment_Date', 'Hospital_discharged_from',
       'Hospital_Admit_Date', 'Admit_weight', 'Diagnosis_1', 'Diagnosis_2',
       'Acute_or_chronic', 'AICD', 'status', 'discharge', 'to_which_hospital',
       'reason_for_dc', 'discharge_date', 'cardiac_related', 'Chain_link_x',
       'weight', 'patient_weight_date', 'This_Weight_Change',
       'Weight_Change_Since_Admit', 'BNP_date', 'BNP', 'This_BNP_Change',
       'Archive', 'labs_date', 'BUN', 'cr', 'Sodium', 'Potasium', 'Mg',
       'facility_Link', 'Chain_link_y', 'Hospitals', 'This_CR_Change']
cardiac.columns
archive.columns
