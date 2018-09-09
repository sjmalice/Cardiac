import gspread
import pandas as pd
from oauth2client import file, client, tools
import numpy as np

# Get credentials
# credentials='/Users/sophiegeoghan/Desktop/MtSinai/API_work/MtSinai-3008a21e1f27.json'
credentials='/Users/sophiegeoghan/Desktop/MtSinai/API_work/credentials_writer.json'

store = file.Storage(credentials)
creds = store.get()
gc = gspread.authorize(creds)

# %% load Google Sheet by Key Name
# opens the Cardiac_M sheet from our google drive:
gs = gc.open_by_key('1qRj0DHYNODEhMZGv1CGBIAgPJQjiBtKuHE68js8dS3A')
gs.worksheets()

# %%

def gsheet2pandas(gsheet):
    """ Convers Google Sheet data from Gspread package to a Pandas
    dataframe """
    header=gsheet.row_values(1) # for some reason this package indexes at 1
    df = pd.DataFrame(columns=header)
    all_records=gsheet.get_all_records()
    for row in np.arange(len(gsheet.get_all_values())-1):
        # print(row)
        tmp_dict=all_records[row]
        # tmp_row = np.array(gsheet.row_values(row)).reshape(1,len(header))
        tmp_df = pd.DataFrame(tmp_dict, index=[row])
        df = pd.concat([df,tmp_df], axis=0)
    print('Google Sheet of size '+str(df.shape)+' successfully loaded')
    return df

def gExcel2pdDict(gexcel,sheet_names):
    """
    from Gspread Google Sheet Object, load all the specified tabs as
    a Python dictionary, similar to Pandas from_excel
    Uses gsheet2pandas function
    """
    all_dict={}
    for st in sheet_names:
        tmp_st=gexcel.worksheet(st)
        tmp_df=gsheet2pandas(tmp_st)
        all_dict[st]=tmp_df
        print('Loaded '+str(st)+' successfully')
    return all_dict
test_st_names=['patients','patient_enrollment_records']
cp=gExcel2pdDict(gs,test_st_names)

cp['patients'].head(10)
df=cp['patients']

# %% Adding column of our predictions to the Google sheet
worksheet=gs.worksheet('patients')
dir(worksheet)

worksheet.add_cols(1)
