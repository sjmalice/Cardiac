import pandas as pd
import numpy as np
import gspread
from gspread import Cell
from oauth2client import file, client, tools
import datetime
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

    response_df['OtherOutputOption']=response_df.Probability.apply(lambda x: "Model predicts {}% chance of hospital readmission".format(np.around(x*100.0,4))
    response_df['Output']=response_df.class_predict.apply(lambda x: "Patient doing fine." if x==1 else "Not good.")
    response_df['Row']=response_df.patient_link.apply(lambda x: df[df['patient_id']==x].index[0])

    report_column=ws.find("Report").col
    cell_list=create_cell_list(ws,response_df['patient_link'].tolist(),report_column,response_df['OtherOutputOption'].tolist(),df)
    ws.update_cells(cell_list)
    print('Successfully updated the Report Column of the google spread sheet')

def update_todays_model(ws,cnf_matrix,accuracy,precision):
    """
    ws: Model_Eval sheet!!
    Updates todays confusion matrix and accuracy ahove the history
    """
    norm_cnf=np.around(cnf_matrix/np.sum(cnf_matrix),4)
    updateCells=[create_Cell(4,2,np.around(accuracy,4)),
                    create_Cell(5,2,np.around(precision,4)),
                    create_Cell(8,2,norm_cnf[0][0]),
                    create_Cell(9,2,norm_cnf[1][0]),
                    create_Cell(8,3,norm_cnf[0][1]),
                    create_Cell(9,3,norm_cnf[1][1])]
    ws.update_cells(updateCells)
    print("Updated the new accuracy etc from today's model")

def create_Cell(row,col,value):
    cell = Cell(row, col)
    cell.value = value
    return cell

def add_to_model_history(ws,cnf_matrix,best_score,precision,thresh_accuracy,thresh_precision):
    """
    adds Confusion Matrix and Accuracy to the ModelEval Worksheet
    Mutating Function. This will update the google sheet!!!
    """
    ws_values=ws.get_all_values()

    df=pd.DataFrame(ws_values)
    date=datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")
    norm_cnf=np.around(cnf_matrix/np.sum(cnf_matrix),4)
    np.hstack(cnf_matrix)
    model_history=np.hstack([date,np.around(best_score,4),np.around(precision,4),np.around(thresh_accuracy,4),np.around(thresh_precision,4),np.hstack(norm_cnf)])
    rowToUpdate=df.shape[0]+1
    cell_list = []
    for i in range(len(model_history)):
        colToUpdate=i+1
        cellToUpdate = Cell(rowToUpdate, colToUpdate)
        cellToUpdate.value = model_history[i]
        cell_list.append(cellToUpdate)
    ws.update_cells(cell_list)
    print("Adding today's model stats to history")
