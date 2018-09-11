import pandas as pd
import numpy as np
import re

import seaborn as sns

from Clean_Fun import *
# %%

#getting the columns that I want out of it
cpa_sheets_load=['patient_enrollment_records', 'patient weights', 'patient BNP',
     'patient BP']
#loading in the data
numerical_data= pd.read_excel('C:/Users/Shani Fisher/Documents/Bootcamp/Capstone/data/Cardiac Program_Archive.xlsx', sheet_name=cpa_sheets_load)
numerical_data.keys()

#Facility_Links
facility=numerical_data['patient_enrollment_records']
numerical_data['patient_enrollment_records']
facility['facilities_link'].value_counts()
facility['facilities_link']=facility['facilities_link'].fillna(value='None')

#dummify the vaiables
pd.get_dummies(facility['facilities_link'])

#not sure if have to make them intigers after they are dummified but if so this is the formula
facility['facilities_link'] = facility['facilities_link'].astype(int)

#This is the histogram plot
sns.distplot(facility['facilities_link'])



#Now working with weight
patient_weight=numerical_data['patient weights']

#something is not turning into a string so here is a way to try and figure it out
def lower_errors(df_df, col_name='weight'):
    try:
        return df_df[col_name].apply(lambda x: x.str.extract('(\d+)'))
    except:
        print(traceback.format_exc())

lower_errors(df_df = patient_weight)

#This is a function to clean weight column and I'm not sure what else at this moment
def clean_weights(df_weights,weight_col='weight',fill_na='0'):
    """df_weghts=the dataframe and which page the column you want is onself.
       weight_col=which column in the dataframe you would like
       This function takes all the numbers from te colum and then plots them on a histogram
    """
    df_weights[weight_col]=patient_weight[weight_col].astype(str)
    df_weights[weight_col]=df_weights[weight_col].str.replace('n/a', '0')
    df_weights[weight_col]=df_weights[weight_col].str.replace(',', '.')
    #this has not worked yet but still trying to get it to work
    df_weights[weight_col].apply(lambda x: weight_col.str.extract('(\d+)'))
    #Did not try this out so not 100% sure it will work but it should based on my other cleaning
    df_bnp[bnp_col]=pd.to_numeric(df_bnp[bnp_col]).round(0).astype(int)
    sns.distplot(df_weights[weight_col])
    # lots of cleanings
    return df_weights

clean_weights(df_weights=)

present_weight=clean_weights(df_weights=numerical_data['patient_weight'])
addmitted_weight=clean_weights(df_weights=numerical_data['patient_enrollment_records'], weight_col='Admit_weight')

#to get weight change I would minues the two things above as
present_weight-addmitted_weight


#Now working on BNP
B_N_P=numerical_data['patient BNP']

#I am trying to replace anything that starts with a c and make it a 0 because come of it has
#writing instead of numbers
#That really should be all the issues there
def clean_BNP(df_bnp, bnp_col='BNP', fillna='0'):
    """Some of the columns have cancelled so getting rid of that
        I am sure that there is an easier way to do this...just not fully sure how!
    """
    #making it a string
    df_bnp[bnp_col]=df_bnp[bnp_col].astype(str)
    #running regex on it so everything is a number
    df_bnp[bnp_col]=df_bnp[bnp_col].replace(['^cancelled'], ['0'], regex=True)
    df_bnp[bnp_col]=df_bnp[bnp_col].replace(['^can'], ['0'], regex=True)
    df_bnp[bnp_col]=df_bnp[bnp_col].replace(['^c patient'], ['0'], regex=True)
    df_bnp[bnp_col]=df_bnp[bnp_col].replace(['^c, patient refused'], ['0'], regex=True)
    df_bnp[bnp_col]=df_bnp[bnp_col].replace(['^not visible in Visual'], ['0'], regex=True)
    #fill nan with a 0
    df_bnp[bnp_col]=df_bnp[bnp_col].str.replace('nan', '0')
    #changing it back into an intiger and then making a histogram with it
    df_bnp[bnp_col]=pd.to_numeric(df_bnp[bnp_col]).round(0).astype(int)
    sns.distplot(df_bnp[bnp_col])

    return(df_bnp)
#This works with both BNP change and BNP now




#Now working with Heart rate and blood pressure...not really blood pressure because we dont know
#how to understand it yet so like how it should be
bp=numerical_data['patient BP']
#could put these all in a function...
resting_hr=resting_hr.fillna(value='0')
resting_hr.str.replace('n/a', '0', regex=False)
resting_hr = resting_hr.iloc[1:]
sns.distplot(resting_hr)
