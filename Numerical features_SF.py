import pandas as pd
import numpy as np
import re

import seaborn as sns

from Clean_Fun import *

# %%

%matplotlib inline
import matplotlib.pyplot as plt


cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
     'patient BP']

#reading in the dataset
numerical_data= pd.read_excel('Data/Cardiac Program_Archive.xlsx', sheet_name=cpa_sheets_load)
numerical_data.keys()

#getting out the facility's so I could run more code on them
facility=numerical_data['patient_enrollment_records']
numerical_data['patient_enrollment_records']
facility['facilities_link'].value_counts()
facility['facilities_link']=facility['facilities_link'].fillna(value='None')

pd.get_dummies
facility=facility.replace({'facilities_link' :{'None':0, 'SdLpKt4S' :1, 'uqmkvwUs': 2, 'voNR0TZJ': 3, '2ueB2F7g':4, '5nKZ993n': 5, 'BodshTSC':6, 'Fy1r9IXM':7, 'BCmqGUNF':8, 'L1V04aB0':9, 'eWM9e2x5':10, 'US4llDDz':11, '71ADiKaS':12, 'mB0G57bu':13, 'kIUZIzRp':14, 'KYzNhByH':15, 'Xxk5Yn3E':16}})
#change to intiger
facility['facilities_link'] = facility['facilities_link'].astype(int)
#hisogram plot
sns.distplot(facility['facilities_link'])

def clean_weights(df_weights,weight_col='weight',fill_na='0'):
    """description here. of input and output
    """
    df_weights[weight_col]=patient_weight[weight_col].astype(str)
    # lots of cleanings
    return df_weights

clean_weights(fill_na='33')

patient_weight.keys()
#patient_weight['weight']
patient_weight=numerical_data['patient weights']
weight=patient_weight['weight']
patient_weight['weight']=patient_weight['weight'].astype(str)
patient_weight['weight']=
patient_weight['weight']=patient_weight['weight'].astype(str)

type(patient_weight['weight'][1])
#patient_weight['weight'].str.replace('\W', '')

re.findall(r'/^[0-9]+$/',patient_weight['weight'])
patient_weight['weight']=patient_weight['weight'].fillna(value='0')
patient_weight['weight']=patient_weight['weight'].str.replace('n/a', '0')
patient_weight['weight'].isna().sum()
s.str.extract(r'([ab])(\d)')
re.findall(r"Test([\d.]*\d+)", patient_weight['weight'])
patient_weight['weight'].str.findall('(\d+)') #it decides that it is all an empty string
#gives e/t NA's
patient_weight['weight'].str.extract('(\d+)')#.astype(str)


''.join(e for e in patient_weight['weight'] if e.isalnum())
patient_weight['weight']=patient_weight['weight'].str.replace('?', '')
patient_weight['weight']=patient_weight['weight'].str.replace(',', '.')
patient_weight['weight']=patient_weight['weight'].str.replace(' ()', '')
patient_weight['weight']=patient_weight['weight'].str.replace('()', ' ')
patient_weight['weight']=patient_weight['weight'].str.replace(' ', '')
re.sub('\ |\?|\.|\!|\/|\,|\:', '', patient_weight['weight'])



pd.to_numeric(patient_weight['weight'], downcast='signed')
patient_weight['weight'].astype(int)
type(patient_weight['weight'])

patient_weight['weight']=pd.to_numeric(patient_weight['weight']).round(0).astype(int)



bp=numerical_data['patient BP']
resting_hr=bp['resting_HR']
resting_hr=resting_hr.fillna(value='0')
resting_hr.str.replace('n/a', '0', regex=False)
resting_hr = resting_hr.iloc[1:]
sns.distplot(resting_hr)

resting_bp=bp['resting_BP']
resting_bp=resting_bp.fillna(value='0')
resting_bp = resting_bp.iloc[1:]

first_numbers=resting_bp.str[:3]
''.join(e for e in first_numbers if e.isalnum())
first_numbers=first_numbers.str.replace('/', '')
first_numbers=first_numbers.str.replace('(', '')
first_numbers=first_numbers.str.replace('', '9')
first_numbers=first_numbers.str.replace('n/a', '0')
first_numbers=first_numbers.fillna(value='0')
first_numbers.astype(int)


last_numbers=resting_bp.str[-2:]
last_numbers=last_numbers.str.replace('/', '')
last_numbers=last_numbers.str.replace('(', '')
last_numbers=last_numbers.str.replace(')', '')
last_numbers=last_numbers.str.replace('P', '')
last_numbers=last_numbers.str.replace('', '9')
last_numbers=last_numbers.str.replace('n/a', '0')
last_numbers=last_numbers.fillna(value='0')
last_numbers.astype(int)
first_numbers/last_numbers
