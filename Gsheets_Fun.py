import pandas as pd
import numpy as np
import gspread
from gspread import Cell
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

def gExcel2pdDict(live_key,creds,sheet_names):
    """
    from Gspread Google Sheet Object, load all the specified tabs as
    a Python dictionary, similar to Pandas from_excel
    Uses gsheet2pandas function
    """
    file = gspread.authorize(creds)
    print("Authenticated with Google")
    gexcel = file.open_by_key(live_key)
    all_dict={}
    for st in sheet_names:
        tmp_st=gexcel.worksheet(st)
        tmp_df=gsheet2pandas(tmp_st)
        all_dict[st]=tmp_df
        print('Loaded '+str(st)+' successfully')
    return all_dict

def create_cell_list(ws,patient_link,colToUpdate,output,df):
    """
    creates a list of cells to update all at once with the call
    worksheet.update_cells(cell_list) from Gspread
    Use as: cell_list=create_cell_list(ws,test_df['patient_link'].tolist(),
                            report_column,test_df['Output'].tolist())
            then:
            ws.update_cells(cell_list)
    """
    cell_list = []
    for i in range(len(patient_link)):
        # print(i)

        cellLookup = df[df['patient_id']==patient_link[i]].index[0]#ws.find(patient_link[i])
        # cellToUpdate = ws.cell(cellLookup+2, colToUpdate)
        cellToUpdate = Cell(cellLookup+2, colToUpdate)
        cellToUpdate.value = output[i]
        cell_list.append(cellToUpdate)
    return cell_list

def upload_predictions(response_df,ws):
    """
    response_df: a df with patient links and predictions from our model
    ws: gspread worksheet object of patients
    Mutating Function. This will update the google sheet!!!
    """
    df=gsheet2pandas(ws)

    response_df=read_pkl('response_df.pkl')
    response_df['Output']=response_df.predictions.apply(lambda x: "Patient doing fine." if x==1 else "Not good.")
    response_df['Row']=response_df.patient_link.apply(lambda x: df[df['patient_id']==x].index[0])

    report_column=ws.find("Report").col
    cell_list=create_cell_list(ws,response_df['patient_link'].tolist(),report_column,response_df['patient_link'].tolist(),df)
    ws.update_cells(cell_list)
    print('Successfully updated the Report Column of the google spread sheet')
