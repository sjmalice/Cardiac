# Data merging functions

import pandas as pd
import numpy as np
import re
from functools import reduce
from Clean_Fun import *

def sheet_merge(file_path, sheet_dict, date_col_list):
    """
    With a path to an Excel doc as input, joins (on patient_link) only
    those sheets and columns specified in a dictionary and
    returns a combined pandas DataFrame.

    Keyword Arguments
    -----------------
    file_path -- File path for the Excel spreadsheet
    sheet_dict -- Dictionary with sheet names (case sensitive) as keys and a list of column names
    as values for each sheet (i.e. {sheet1_name: [col1, col2,...coln], sheet2_name: [col1,...]})
    date_col_list -- A list of date column names on which to filter by most recent measurement date
    """
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
    for sheet_name, col in zip(sheets, list(map(str.lower, date_col_list))):
        init_len = len(df_dict[sheet_name])
        df_dict[sheet_name] = choose_most_recent(df_dict[sheet_name], col)
        print('Sheet \"{}\" reduced from {} rows to {} rows'.format(sheet_name, init_len, len(df_dict[sheet_name])))

    # full_df = reduce(lambda x, y: pd.merge(x, y, on = 'patient_link'), list(df_dict.values()))


########################
####### Testing ########
########################

sheet_dict = {'patient_enrollment_records': ['patient_link', 'facilities_link', 'Enrollment_Date', 'admit_weight'],\
             'patient weights': ['patient_link', 'patient_weight_date', 'weight', 'weight_change_since_admit'],\
             'patient BNP': ['patient_link', 'bnp_date', 'bnp', 'this_bnp_change'],\
             'Cardiac_Meds': ['patient_link', 'Cardiac_Meds_Date', 'ACE', 'BB', 'Diuretics', 'Anticoagulant', 'Ionotropes', 'Other cardiac meds'],\
             'patient labs': ['patient_link', 'labs_date', 'BUN', 'cr', 'Sodium', 'Potasium', 'Mg'],\
             'patient BP': ['patient_link', 'bp_date', 'resting_hr', 'systolic', 'diastolic', 'resting_bp', 'mets_base_line', 'mets']}

sheets = list(sheet_dict.keys())
df_dict = pd.read_excel('Data/Cardiac Program_Archive.xlsx', sheet_name=sheets)

for sheet_name in sheets:
    df_dict[sheet_name].columns = [x.lower() for x in df_dict[sheet_name].columns]
    df_dict[sheet_name] = df_dict[sheet_name][list(map(str.lower, sheet_dict[sheet_name]))]
    print('Sheet name: \"{}\"'.format(sheet_name))
    print('Retained columns: {}\n'.format(list(df_dict[sheet_name].columns)))

date_col_list = ['enrollment_date', 'patient_weight_date', 'bnp_date', 'cardiac_meds_date', 'labs_date', 'bp_date']

for sheet_name, col in zip(sheets, list(map(str.lower, date_col_list))):
    init_len = len(df_dict[sheet_name])
    df_dict[sheet_name] = choose_most_recent(df_dict[sheet_name], col)
    print('Sheet \"{}\" reduced from {} rows to {} rows'.format(sheet_name, init_len, len(df_dict[sheet_name])))

for sheet_name in sheets:
    print(len(df_dict[sheet_name]))
for sheet_name in sheets:
    print(df_dict[sheet_name]['patient_link'].dtype)
reduce(lambda x, y: pd.merge(x, y, on='patient_link'), list(df_dict.values()))

temp_df = df_dict['patient_enrollment_records']
sheets_no_first = ['patient weights', 'patient BNP', 'Cardiac_Meds', 'patient labs', 'patient BP']
