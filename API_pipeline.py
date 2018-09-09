import pandas as pd
import numpy as np
import json
import gspread
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from Clean_Fun import *
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# %%

scope = ['https://www.googleapis.com/auth/spreadsheets']
credentials = ServiceAccountCredentials.from_json_keyfile_name(#json_key,scope)
    '/Users/sophiegeoghan/Desktop/MtSinai/API_work/MtSinai-3008a21e1f27.json', scope)

file = gspread.authorize(credentials) # authenticate with Google

gs = file.open_by_key('1qRj0DHYNODEhMZGv1CGBIAgPJQjiBtKuHE68js8dS3A')

# %%

test_st_names=['patients','patient_enrollment_records']
cp=gExcel2pdDict(gs,test_st_names)

df=cp['patient_enrollment_records']
test=find_unique_diag(df['Diagnosis_1'])
diag_dumm_df=dummify_diagnoses(df,test)

# %% Adding worksheet of our dummification to the Google sheet

worksheet = gs.add_worksheet(title="Diagnoses2", rows=str(diag_dumm_df.shape[0]), cols=diag_dumm_df.shape[1])
set_with_dataframe(worksheet, df)
