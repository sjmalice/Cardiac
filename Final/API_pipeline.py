%load_ext autoreload
%autoreload 2
# %%
import pandas as pd
import numpy as np
import json
import gspread
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from Clean_Fun import *
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import pickle
from API_data_merge import *
from Gsheets_Fun import *
import time
from Clean_Fun import *
from Meta_fun import *
# %%

live_sheet_pkl_path='pickle_jar/live_sheets.pkl'
archive_sheet_pkl='pickle_jar/archive_sheets.pkl'
datecol_pkl_path='pickle_jar/datecols.pkl'

scope = ['https://www.googleapis.com/auth/spreadsheets']
creds = ServiceAccountCredentials.from_json_keyfile_name(#json_key,scope)
    '/Users/sophiegeoghan/Desktop/MtSinai/API_work/MtSinai-3008a21e1f27.json', scope)

live_key_old = '1cN9pczJq3ZxThz0kiYRfrB8StfzPg5hCnHqFLiDpxMU'
live_key='12J8CfUpexmdTeu5gsybh1kchD57B7r702HrM-M-j7u0'
archive_key = '1_p6EyLT1E8ExgTgm7_n0ctGwekrAzzzjYcaejmWTj9o'

# %%
start_time = time.time()
df = gs_sheet_merge(live_key,archive_key, live_sheet_pkl_path, archive_sheet_pkl, datecol_pkl_path,creds)

print('-'*50)
print("--- %s seconds ---" % (time.time() - start_time))
print('Successfully pulled new data from Google API')
df.to_csv('SuccessArchivetoo2.csv')

df=pd.read_csv('SuccessArchivetoo2.csv',index_col=0)
# %% Cleaning and Modelling
from clean_model import *
# df.sample(5)

df_full = meta_clean(df)

return_df, accuracy, precision, cnf_matrix, thresh_accuracy, thresh_precision, cnf_thresh_matrix=logistic_model(df_full,threshold=0.6)

# %% Adding worksheet of our dummification to the Google sheet

# %%
print("Reauthorizing with Google.")

file = gspread.authorize(creds)

gs = file.open_by_key(live_key)

ws=gs.worksheet('patients')
eval_ws=gs.worksheet("Model_Eval")

update_todays_model(eval_ws,cnf_matrix,accuracy,precision)

add_to_model_history(eval_ws,cnf_matrix,accuracy,precision,thresh_accuracy,thresh_precision)
# return_df.columns=['patient_link','enrollId','class_predict','Probability']
upload_predictions(return_df,ws)
