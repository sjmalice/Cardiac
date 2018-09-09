import pandas as pd
import numpy as np
import gspread
from oauth2client import file, client, tools

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
