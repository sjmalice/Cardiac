# Data merging functions
import pandas as pd
import numpy as np
import pickle
import re
from functools import reduce
from Clean_Fun import *

def archive_sheet_merge(file_path, sheet_pkl, datecol_pkl):
    """
    With a path to archive Excel doc as input, joins (on patient_link) only
    those sheets and columns specified in a pickle dictionary and returns a combined pandas DataFrame.

    Keyword Arguments
    -----------------
    file_path -- File path for the Excel spreadsheet
    sheet_pkl -- Path to pickle containing dictionary with sheet names (keys, case sensitive) and a list of column names (values)
    for each sheet (i.e. {sheet1_name: [col1, col2,...coln], sheet2_name: [col1,...]})
    datecol_pkl -- Path to pickle containing dictionary of sheet names (keys, case sensitive) and date column names (values) on which to
    filter by most recent measurement date. Only one column name per sheet (i.e. {sheet1_name: date_col_name1, ...})
    """
    # Load sheet dictionary from pickle
    with open(sheet_pkl, 'rb') as f:
        sheet_dict = pickle.load(f)

    # Load date_col dictionary from pickle
    with open(datecol_pkl, 'rb') as f:
        date_col_dict = pickle.load(f)

    # Load the specified sheets into a dictionary of sheets as keys and dataframes as values
    sheets = list(sheet_dict.keys())
    df_dict = pd.read_excel(file_path, sheet_name=sheets)

    # Convert all column names to lowercase for consistancy
    # For each sheet dataframe, keep only the specified columns
    for sheet_name in sheets:
        df_dict[sheet_name].columns = [x.lower() for x in df_dict[sheet_name].columns]
        df_dict[sheet_name] = df_dict[sheet_name][list(map(str.lower, sheet_dict[sheet_name]))]
        print('Sheet name: \"{}\"'.format(sheet_name))
        print('Retained columns: {}\n'.format(sheet_dict[sheet_name]))

    # Filter rows with most recent measurement date to remove duplicate patients
    # If date column not given for particular sheet name, skip that sheet
    for sheet_name in list(date_col_dict.keys()):
        if sheet_name in list(df_dict.keys()):
            init_len = len(df_dict[sheet_name])
            df_dict[sheet_name] = choose_most_recent(df_dict[sheet_name], date_col_dict[sheet_name].lower())
            print('Sheet \"{}\" reduced from {} rows to {} rows'.format(sheet_name, init_len, len(df_dict[sheet_name])))
        else:
            print('Sheet \"{}\" did not have a date column to filter by'.format(sheet_name))
            continue

    return reduce(lambda x, y: pd.merge(x, y, on='patient_link', how='outer'), list(df_dict.values()))

def live_sheet_merge(file_path, sheet_pkl, datecol_pkl):
    """
    With a path to live Excel doc as input, joins (on patient_link) only
    those sheets and columns specified in a pickle dictionary and returns a combined pandas DataFrame.

    Keyword Arguments
    -----------------
    file_path -- File path for the Excel spreadsheet
    sheet_pkl -- Path to pickle containing dictionary with sheet names (keys, case sensitive) and a list of column names (values)
    for each sheet (i.e. {sheet1_name: [col1, col2,...coln], sheet2_name: [col1,...]})
    datecol_pkl -- Path to pickle containing dictionary of sheet names (keys, case sensitive) and date column names (values) on which to
    filter by most recent measurement date. Only one column name per sheet (i.e. {sheet1_name: date_col_name1, ...})
    """
    # Load sheet dictionary from pickle
    with open(sheet_pkl, 'rb') as f:
        sheet_dict = pickle.load(f)

    # Load date_col dictionary from pickle
    with open(datecol_pkl, 'rb') as f:
        date_col_dict = pickle.load(f)

    # Load the specified sheets into a dictionary of sheets as keys and dataframes as values
    sheets = list(sheet_dict.keys())
    df_dict = pd.read_excel(file_path, sheet_name=sheets)

    df_dict['patients'].rename(columns={'patient_id': 'patient_link'}, inplace=True)
    sheet_dict['patients'][0] = 'patient_link'

    # Convert all column names to lowercase for consistancy
    # For each sheet dataframe, keep only the specified columns
    for sheet_name in sheets:
        df_dict[sheet_name].columns = [x.lower() for x in df_dict[sheet_name].columns]
        df_dict[sheet_name] = df_dict[sheet_name][list(map(str.lower, sheet_dict[sheet_name]))]
        print('Sheet name: \"{}\"'.format(sheet_name))
        print('Retained columns: {}\n'.format(sheet_dict[sheet_name]))

    # Filter rows with most recent measurement date to remove duplicate patients
    # If date column not given for particular sheet name, skip that sheet
    for sheet_name in list(date_col_dict.keys()):
        if sheet_name in list(df_dict.keys()):
            init_len = len(df_dict[sheet_name])
            df_dict[sheet_name] = choose_most_recent(df_dict[sheet_name], date_col_dict[sheet_name].lower())
            print('Sheet \"{}\" reduced from {} rows to {} rows'.format(sheet_name, init_len, len(df_dict[sheet_name])))
        else:
            print('Sheet \"{}\" did not have a date column to filter by'.format(sheet_name))
            continue

    return reduce(lambda x, y: pd.merge(x, y, on='patient_link', how='outer'), list(df_dict.values()))
