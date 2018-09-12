import pandas as pd
import numpy as np
from data_merge import *
from Clean_Fun import *

# NOTE have to use remove_invalid_rows() inside ALex's function,
# before we remove patient name

# %% Load dataset

file_path='Data/Cardiac Program_M2.xlsx'
sheet_pkl='pickle_jar/live_sheets.pkl'
datecol_pkl='pickle_jar/live_datecols.pkl'
df=live_sheet_merge(file_path, sheet_pkl, datecol_pkl)

# df=pd.read_csv('cardiac.csv')
# df=df.drop('Unnamed: 0',axis=1)

# %% Dropping cardiac related, test patients, determing Response Value

# let's talk about removing cardiac related - maybe we pursue multiple
# classification model and keep cardiac related in our dataset?
# remove_cardiac_unrelated(df)

train_df,test_df=train_test_split_sg(df)

df=train_df.copy() # for now

# %% Clean effusion rate

df['ef']=df['ef'].apply(lambda x: clean_EF_rows(x))

# Clean Blood Pressure rows

df['diastolic']=df.apply(lambda row: clean_diastolic_columns(
    row['diastolic'],row['resting_bp'],col_type='di'),axis=1)
df['systolic']=df.apply(lambda row: clean_diastolic_columns(
    row['systolic'],row['resting_bp'],col_type='sys'),axis=1)

# Dummify the diagnoses
uniq_diag=find_unique_diag(df.diagnosis_1)
dummy_df_diag=dummify_diagnoses(df,uniq_diag,diagnosis_col='diagnosis_1')
df.drop('diagnosis_1',axis=1,inplace=True)
df=df.merge(dummy_df_diag,on='patient_link',how="inner")

# Clean Meds and aicd
# acute or chronic

med_aicd_clean(df,'ace', 0)
med_aicd_clean(df,'bb', 0)
med_aicd_clean(df,'diuretics', 0)
med_aicd_clean(df,'anticoagulant', 0)
med_aicd_clean(df,'ionotropes', 0)
med_aicd_clean(df,'aicd', 0)

weight_dur_age_clean(df,dur_na=-999999,age_na=-99.,weight_perc_cutoff=0.2)

# %%
