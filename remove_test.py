import numpy as np
import pandas as pd


# %%

cpa=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name=['patient_enrollment_records'])
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=['patients'])

df=cpa['patient_enrollment_records']
df.columns

# find the index of invalid rows
ind_inv=df.loc[df['patient_link'].apply(lambda x: True if len(str(x))<3 else False)].index
# print and remove them
for i in ind_inv:
    print("removing invalid row: \n")
    try:
        print(df.iloc[i][['patient_link','Enrollment_Date','patient_name','create_user']])
    except:
        print(df.iloc[i]['patient_link'])
    print('-'*50)
    df.drop([i],axis=0,inplace=True)

df=df.reset_index()

# find observations from Test
ind_test=df.loc[df['create_user']=='multitechvisions@gmail.com'].index#[['patient_link','Enrollment_Date','patient_name','create_user']]
ind_test
df.iloc[ind_test]['create_user']

df.sample(10)
print("removing multitechvisions test rows: \n")
try:
    print(df.iloc[ind_test][['patient_link','Enrollment_Date','patient_name','create_user']])
except:
    print(df.iloc[ind_test]['patient_link'])
print('-'*50)
df.drop(df.loc[df['create_user']=='multitechvisions@gmail.com'],axis=1,inplace=True)
df.reset_index(drop=True)

df.iloc[ind]




df=df.drop()
