import pandas as pd
import numpy as np

import seaborn as sns

from Clean_Fun import *

'''
All sheets from CPA:
['patient_enrollment_records', 'Settings', 'patient weights', 'patient BNP',
'Cardiac_Meds', 'patient labs', 'patient BP', 'notes',
"Old notes (Don't fit structure)"]

To do:
If we have time, clean up Notes and Old Notes

 '''

cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
    'Cardiac_Meds','patient labs', 'patient BP', 'notes',"Old notes (Don't fit structure)"]
cpa=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name=cpa_sheets_load)
cpa.keys()

# %% Enrollment records
# Attempt to simply separate Patients that have been discharged or not:
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
df_train=df_train[keep_columns]
df_train.patient_link.unique().shape

enroll_date=choose_most_recent(df_train,'Enrollment_Date')

# %% Weight Archive
# we could use df_train patient links to always separate the dataset
weight=cpa['patient weights']
weight.shape
weight_columns=['patient_link', 'patient_weight_date', 'weight',
       'Next_weight_due', 'editcount', 'edit_time_stamp',
       'facility_Link', 'Chain_link', 'Hospitals', 'This_Weight_Change',
       'Weight_Change_Since_Admit']

weight=weight.loc[weight['patient_link'].apply(lambda x: True if (x in train_patients.values) else False)]
weight.shape
weight=weight[weight_columns]
tmp=weight.loc[weight.patient_link=='W6X7LdR5']
tmp.loc[tmp.patient_weight_date==max(tmp.patient_weight_date)]

# we don't have the weight for every patient

import matplotlib.pyplot as plt
weight.patient_link.value_counts().plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
plt.title('Number of records of Patient weights')
plt.xlabel('Number of times 1 patient was weighed')
plt.ylabel('Frequency')
weight.dtypes
weight=weight[['patient_link','weight','patient_weight_date', 'This_Weight_Change','Weight_Change_Since_Admit']]
# malpractice['DateTime']=list(map(lambda x:pd.to_datetime(x),malpractice.date))
print(weight.patient_weight_date[0])
print(weight.patient_weight_date[1]-weight.patient_weight_date[0])

# row,.=weight.shape

weight.patient_weight_date=weight.patient_weight_date.apply(lambda x:pd.to_datetime(x,errors='coerce'))
max(weight.patient_weight_date)

#to complete later to coerce typos into 2017 instead of 0207 :/
def handle_date_typos(x):
    try:
        return pd.to_datetime(x)
    except:
        weight.loc[weight.patient_weight_date==x].index
# %%

weight_nodupes =choose_most_recent(weight,'patient_weight_date')

# %%

# %% BNP
bnp=cpa['patient BNP']
bnp.dtypes
bnp.columns
bnp=bnp[['Patient_link','BNP_date', 'BNP','This_BNP_Change', 'Archive']]
bnp.columns=['patient_link','BNP_date', 'BNP','This_BNP_Change', 'Archive']
bnp.BNP_date=bnp.BNP_date.apply(lambda x:pd.to_datetime(x,errors='coerce'))
bnp=bnp.loc[bnp['patient_link'].apply(lambda x: True if (x in train_patients.values) else False)]

# bnp.patient_link.value_counts().plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
bnp_nodupes=choose_most_recent(bnp,'BNP_date')

# %% Cardiac meds are next but that would involved text cleaning I'll leave for now
plab=cpa['patient labs']
plab.columns
plab=plab[['labs_date', 'BUN', 'cr', 'Sodium', 'Potasium', 'Mg',
       'patient_link','facility_Link', 'Chain_link', 'Hospitals', 'This_CR_Change']]
plab=plab.loc[plab['patient_link'].apply(lambda x: True if (x in train_patients.values) else False)]

labs_nodupes=choose_most_recent(plab,'labs_date')

all_nodupes=pd.merge(enroll_date,weight_nodupes,on='patient_link',how='outer')
all_nodupes=pd.merge(all_nodupes,bnp_nodupes,on='patient_link',how='outer')
all_nodupes.shape
all_nodupes=pd.merge(all_nodupes,labs_nodupes,on='patient_link',how='outer')
all_nodupes.shape
all_nodupes.to_csv('ArchiveClean.csv')
# test=all_nodupes.columns
# heat=sns.heatmap(all_nodupes[test].isnull(), cbar=False)
# %%

plt.subplots(figsize=(20,15))
heat=sns.heatmap(all_nodupes.isnull(), cbar=False)
# plt.xticks(rotation=75)
fig=heat.get_figure()
fig.savefig('archive_missingness.png',transparent=True, dpi=400,bbox_inches='tight' ,format='png')

# %%
