# Merge Cardiac and Archive

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

cardiac=pd.read_csv('CleanData/CardiacM_clean.csv')
archive=pd.read_csv('CleanData/ArchiveClean.csv')
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
# fig.savefig('cardiac_merge.png',transparent=True, dpi=400,bbox_inches='tight' ,format='png')
test.sample(10)
# %% Figure out a better merge system
['Date_of_Birth', 'Patient_Gender', 'EF', 'Date of last echo',
       'cardiac_plan', 'patient_link', 'edit_time_stamp',
       'Current_Facility_value', 'Current_Chain_value', 'Report',
       'Hospital_History', 'Archive_Patient', 'Last_Weight', 'Weight_Change',
       'Weight_Change_Since_Admit_x', 'Last_BNP', 'BNP_Change', 'BNP_x',
       'Last_Labs', 'CR_Change', 'Last_Meds', 'Admin Notes', 'special_status',
       'Enrollment_Date_x', 'Hospital_discharged_from_x',
       'Hospital_Admit_Date_x', 'Admit_weight_x', 'Diagnosis_1_x',
       'Diagnosis_2_x', 'Acute_or_chronic_x', 'AICD_x', 'status_x',
       'discharge_x', 'to_which_hospital_x', 'reason_for_dc_x',
       'discharge_date_x', 'cardiac_related_x', 'Enrollment_Active',
       'Chain_link', 'Enrollment_Date_y', 'Hospital_discharged_from_y',
       'Hospital_Admit_Date_y', 'Admit_weight_y', 'Diagnosis_1_y',
       'Diagnosis_2_y', 'Acute_or_chronic_y', 'AICD_y', 'status_y',
       'discharge_y', 'to_which_hospital_y', 'reason_for_dc_y',
       'discharge_date_y', 'cardiac_related_y', 'Chain_link_x', 'weight',
       'patient_weight_date', 'This_Weight_Change',
       'Weight_Change_Since_Admit_y', 'BNP_date', 'BNP_y', 'This_BNP_Change',
       'Archive', 'labs_date', 'BUN', 'cr', 'Sodium', 'Potasium', 'Mg',
       'facility_Link', 'Chain_link_y', 'Hospitals', 'This_CR_Change']
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

duplicate_col=[]
for col in cardiac.columns:
    if col in archive.columns:
        duplicate_col.append(col)
duplicate_col.remove('patient_link')
duplicate_col
test['Weight_Change_Since_Admit_x']
test[duplicate_col[0]+'_x'][1]
c
pat
duplicate_col[c]+'_x'
'Weight_Change_Since_Admit'
test[duplicate_col[c]+'_x'][pat]
test_merge_results=[]
for pat in range(test.shape[0]):
    for c in duplicate_col:
        if test[[duplicate_col[c]+'_x']][pat]==test[[duplicate_col[0]+'_y']][pat]:
            test_merge_results.append(True)
        else:
            test_merge_results.append(False)
cardiac.columns
archive.columns
