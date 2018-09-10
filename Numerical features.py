import pandas as pd
import numpy as np
import re

import seaborn as sns

from Clean_Fun import *

%matplotlib inline
import matplotlib.pyplot as plt


cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
     'patient BP']

#reading in the dataset
numerical_data= pd.read_excel('C:/Users/Shani Fisher/Documents/Bootcamp/Capstone/data/Cardiac Program_Archive.xlsx', sheet_name=cpa_sheets_load)
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



patient_weight.keys()
#patient_weight['weight']
patient_weight=numerical_data['patient weights']
weight=patient_weight['weight']
patient_weight['weight']=patient_weight['weight'].astype(str)
patient_weight['weight']=
patient_weight['weight']=patient_weight['weight'].astype(str)

type(patient_weight['weight'][1472])
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
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])
df.A.str.extract('(\d+)')


patient_weight['weight']=patient_weight['weight'].fillna(value='0')
patient_weight['weight']=patient_weight['weight'].str.replace('n/a', '0')
patient_weight['weight'].apply(lambda x: lower_errors(x)).unique()
patient_weight['weight']=patient_weight['weight'].str.replace(',', '.')
import traceback

patient_weight['weight'].apply(lambda x: x.str.extract('(\d+)'))

def lower_errors(df_df, col_name='weight'):
    try:
        return df_df[col_name].apply(lambda x: x.str.extract('(\d+)'))
    except:
        print(traceback.format_exc())
patient_weight['weight']=list(map(lambda x : str(x) ,patient_weight['weight']))
lower_errors(df_df = patient_weight)


def clean_weights(df_weights,weight_col='weight',fill_na='0'):
    """df_weghts=the dataframe and which page the column you want is onself."""
    """weight_col=which column in the dataframe you would like"""
    """ This function takes all the numbers from te colum and then plots them on a histogram
    """
    df_weights[weight_col]=patient_weight[weight_col].astype(str)
    df_weights[weight_col]=df_weights[weight_col].str.replace('n/a', '0')
    df_weights[weight_col]=df_weights[weight_col].str.replace(',', '.')
    df_weights[weight_col].apply(lambda x: weight_col.str.extract('(\d+)'))
    sns.distplot(df_weights['weight])
    # lots of cleanings
    return df_weights

present_weight=clean_weights(df_weights=numerical_data['patient_weight'])
addmitted_weight=clean_weights(df_weights=numerical_data['patient_enrollment_records'], weight_col='Admit_weight')

#to get weight change I would minues the two things above as

present_weight-addmitted_weight


''.join(e for e in patient_weight['weight'] if e.isalnum())
patient_weight['weight']=patient_weight['weight'].str.replace('?', '')
patient_weight['weight']=patient_weight['weight'].str.replace(',', '.')
patient_weight['weight']=patient_weight['weight'].str.replace(' ()', '')
patient_weight['weight']=patient_weight['weight'].str.replace('()', ' ')
patient_weight['weight']=patient_weight['weight'].str.replace(' ', '')
re.sub('\ |\?|\.|\!|\/|\,|\:', '', patient_weight['weight'])
patient_weight.apply(lambda x: lower_errors(x))



pd.to_numeric(patient_weight['weight'], downcast='signed')
patient_weight['weight'].astype(int)
type(patient_weight['weight'])

patient_weight['weight']=pd.to_numeric(patient_weight['weight']).round(0).astype(int)



bp=numerical_data['patient BP']
resting_hr=bp['resting_HR']

resting.drop(resting.index[[1098,1233]])

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



B_N_P=numerical_data['patient BNP']
B_N_P['BNP']=B_N_P['BNP'].replace(['^cancelled'], ['0'], regex=True)
B_N_P['BNP']=B_N_P['BNP'].replace(['^can'], ['0'], regex=True)
B_N_P['BNP']=B_N_P['BNP'].replace(['^c patient'], ['0'], regex=True)
B_N_P['BNP']=B_N_P['BNP'].replace(['^c, patient refused'], ['0'], regex=True)
B_N_P['BNP']=B_N_P['BNP'].replace(['^not visible in Visual'], ['0'], regex=True)
B_N_P['BNP']=B_N_P['BNP'].astype(str)
B_N_P['BNP']=pd.to_numeric(B_N_P['BNP']).round(0).astype(int)
sns.distplot(B_N_P['BNP'])



B_N_P=numerical_data['patient BNP']

B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].astype(str)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].replace(['^cancelled'], ['0'], regex=True)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].replace(['^can'], ['0'], regex=True)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].replace(['^c patient'], ['0'], regex=True)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].replace(['^c, patient refused'], ['0'], regex=True)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].replace(['^not visible in Visual'], ['0'], regex=True)
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].fillna(value='0')
B_N_P['This_BNP_Change']=B_N_P['This_BNP_Change'].str.replace('nan', '0')
B_N_P['This_BNP_Change']=pd.to_numeric(B_N_P['This_BNP_Change']).round(0).astype(int)
sns.distplot(B_N_P['This_BNP_Change'])
