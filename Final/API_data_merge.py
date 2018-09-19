# Data merging functions
import pandas as pd
import numpy as np
import pickle
from Clean_Fun import *
from API_enrollId import *
from Gsheets_Fun import *

def gs_sheet_merge(live_key, archive_key,live_sheet_pkl_path, archive_sheet_pkl_path,datecol_pkl_path,creds):
    """
    Concatonates pairs of sheets from two different spreadsheets, retaining only the sheets and columns specified
    in pickles. Returns a dictionary with keys as sheet names and values as concatonated dataframes.

    Keyword Arguments
    -----------------
    live_path -- File path for the live data spreadsheet
    archive_path -- File path for the archive data spreadsheet
    live_sheet_pkl_path -- Path to pickle containing dictionary for live spreadsheet with sheet names (keys, case sensitive) and a list
    of column names (values) for each sheet (i.e. {sheet1_name: [col1, col2,...coln], sheet2_name: [col1,...]})
    archive_sheet_pkl_path -- Same as live_sheet_pkl_path but for the archive spreadsheet
    datecol_pkl_path -- Path to pickle containing dictionary of sheets as keys and their associated date columns

    Returns
    -------
    Fully merged dataframe
    """
    # Load live sheet dictionary from pickle
    live_sheet_dict = read_pkl(live_sheet_pkl_path)

    # Load archive sheet dictionary from pickle
    archive_sheet_dict = read_pkl(archive_sheet_pkl_path)

    # Load date columns from pickle
    datecol_dict = read_pkl(datecol_pkl_path)

    # Store sheet dictionary keys as variables and load dataframe dictionaries from spreadsheets
    archive_sheets = list(archive_sheet_dict.keys())
    live_sheets = list(live_sheet_dict.keys())
    archive_df_dict = gExcel2pdDict(archive_key, creds, sheet_names=archive_sheets)
    live_df_dict = gExcel2pdDict(live_key, creds, sheet_names=live_sheets)

    # Rename 'patient_id' column to 'patient_link' for consistency, do the same in live sheet dictionary
    live_df_dict['patients'].rename(columns={'patient_id': 'patient_link'}, inplace=True)
    live_sheet_dict['patients'][0] = 'patient_link'

    # Convert all column names to lowercase for consistancy
    # For each sheet dataframe, keep only the specified columns
    print('Live Sheet:')
    for sheet_name in live_sheets:
        live_df_dict[sheet_name].columns = [x.lower().strip() for x in live_df_dict[sheet_name].columns]
        live_df_dict[sheet_name] = live_df_dict[sheet_name][list(map(str.lower, live_sheet_dict[sheet_name]))]
        print('Sheet name: "{}"'.format(sheet_name))
        print('Retained columns: {}\n'.format(list(live_df_dict[sheet_name].columns)))
    print('\n')

    # Repeat for archive sheets
    print('Archive Sheet:')
    for sheet_name in archive_sheets:
        archive_df_dict[sheet_name].columns = [x.lower().strip() for x in archive_df_dict[sheet_name].columns]
        archive_df_dict[sheet_name] = archive_df_dict[sheet_name][list(map(str.lower, archive_sheet_dict[sheet_name]))]
        print('Sheet name: "{}"'.format(sheet_name))
        print('Retained columns: {}\n'.format(list(archive_df_dict[sheet_name].columns)))
    print('\n')

    # Make sure all date columns have appropriate datetime format, otherwise change to NaT
    for sheet_name in list(datecol_dict.keys()):
        live_df_dict[sheet_name][datecol_dict[sheet_name]] = datetime_fixer(live_df_dict[sheet_name][datecol_dict[sheet_name]].apply(lambda x: str(x)))
        archive_df_dict[sheet_name][datecol_dict[sheet_name]] = datetime_fixer(archive_df_dict[sheet_name][datecol_dict[sheet_name]].apply(lambda x: str(x)))
        print('Date column "{}" in sheet "{}" has been converted to appropriate datatype\n'.format(datecol_dict[sheet_name], sheet_name))

    live_df_dict['patients']['date_of_birth'] = datetime_fixer(live_df_dict['patients']['date_of_birth'].apply(lambda x: str(x)))
    print('Date column "date_of_birth" in sheet "patients" has been converted to appropriate datatype')
    print('\n')
    combined_df_dict = {}

    # Concatonate pairs of sheets and store in new combined_df_dict
    for sheet_name in archive_sheets:
        combined_df_dict[sheet_name] = pd.concat([archive_df_dict[sheet_name], live_df_dict[sheet_name]], ignore_index=True)

    # Add 'patients' sheet to combined_df_dict, because it is not in archive
    combined_df_dict['patients'] = live_df_dict['patients']

    # Add enrollID column to patient_enrollment_records sheet
    enrollId_df = generateEnrollId(combined_df_dict['patient_enrollment_records'])
    enrollId_df['enrollment_date']=enrollId_df.enrollment_date.apply(lambda x: pd.to_datetime(str(x)))
    # Add enrollID column to other sheets and filter most recent date per enrollId
    for sheet_name in archive_sheets:
        combined_df_dict[sheet_name] = addEnrollId(combined_df_dict[sheet_name], datecol_dict[sheet_name], enrollId_df)
        print('Added "enrollID" column to sheet "{}"'.format(sheet_name))
        init_length = len(combined_df_dict[sheet_name])
        # combined_df_dict[sheet_name] = choose_most_recent(combined_df_dict[sheet_name], datecol_dict[sheet_name])
        combined_df_dict[sheet_name] = choose_most_recent(combined_df_dict[sheet_name], datecol_dict[sheet_name])
        print('Reduced sheet "{}" from {} rows to {} rows by filtering most recent dates with least missing\n'.format(sheet_name, init_length, len(combined_df_dict[sheet_name])))
    print('\n')

    # Merge patients sheet with patient_enrollment_records manually because patients does not have an enrollId column
    full_df = pd.merge(combined_df_dict['patients'], combined_df_dict['patient_enrollment_records'], on='patient_link', how='inner')

    # Merge the rest of the dataframes on enrollId
    for sheet_name in archive_sheets:
        if sheet_name != 'patient_enrollment_records':
            init_length = len(full_df)
            combined_df_dict[sheet_name].drop(columns='patient_link', inplace=True)
            full_df = pd.merge(full_df, combined_df_dict[sheet_name], on='enrollId', how='inner')
            print('Merged sheet "{}" with the full dataframe and changed the row number by {}'.format(sheet_name, (len(full_df)-init_length)))
    print('\n')
    print('DATA MERGE IS COMPLETE')
    return full_df
