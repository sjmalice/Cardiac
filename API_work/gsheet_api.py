import gspread
import pandas as pd
from oauth2client import file, client, tools
import numpy as np

credentials='/Users/sophiegeoghan/Desktop/MtSinai/API_work/client_secret.json'

store = file.Storage(credentials)
creds = store.get()

gc = gspread.authorize(creds)

gs = gc.open_by_key(KEYHERE)
st=gs.get_worksheet(12)
# %%

def gsheet2pandas(gsheet):
    """ Convers Google Sheet data from Gspread package to a Pandas
    dataframe """
    header=gsheet.row_values(1) # for some reason this package indexes at 1
    df = pd.DataFrame(columns=header)
    for row in np.arange(len(gsheet.get_all_values())):
        print(row)
        tmp_dict=gsheet.get_all_records()[row]
        # tmp_row = np.array(gsheet.row_values(row)).reshape(1,len(header))
        tmp_df = pd.DataFrame(tmp_dict, index=[row])
        df = pd.concat([df,tmp_df], axis=0)
    return df

df=gsheet2pandas(st)

df
