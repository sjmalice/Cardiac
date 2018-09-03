import pandas as pd
import numpy as np

import seaborn as sns

from Clean_Fun import *

''' All sheets from CP:
 ['Menu', 'Notifications', 'users', 'Hospitals', 'Reports',
'Bundle_Reports', 'chains', 'facilities', 'Statuses', 'BGs', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

Some notes on them:
Sheet30 - Just building the visual for how many days they've been in hospital

To do:
Notes sheet - determine patient_link
'''

sheets_to_load=['Hospitals', 'chains', 'facilities', 'Statuses', 'Sheet30', 'EKGs',
 'patients', 'patient_enrollment_records', 'patient labs', 'patient weights',
 'patient BNP', 'patient BP', 'Cardiac_Meds', 'notes']

# Creates a dictionary with each sheet from the Excel, sheet_name=None means load all sheets
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=sheets_to_load)
cp['patient_enrollment_records'].columns

# this helped me realize that records from the archive, have their Enrollment Records in the not-Archive. So merging is a necessity
cp['patients'].loc[cp['patients']['patient_id']=='W6X7LdR5']['Patient_Name','','Admin Notes']
cp['patient_enrollment_records'].loc[cp['patient_enrollment_records']['patient_link']=='W6X7LdR5']
