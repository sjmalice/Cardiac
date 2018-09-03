# EDA on Mt Sinai Project
import pandas as pd
import numpy as np

patients=pd.read_csv('Cardiac Program_M - patients.csv')
patients.shape
patients.columns

enrollments=pd.read_csv('Cardiac Program_M - patient_enrollment_records.csv')
enrollments.shape
enrollments.columns
enroll_arch=pd.read_csv('Cardiac Program_Archive - patient_enrollment_records.csv')
enroll_arch.shape
merge_on=enroll_arch.columns
allEnroll=pd.merge(enrollments,enroll_arch,on='patient_link',how='outer')
allEnroll['patient_link'].unique().shape
allEnroll.loc[allEnroll['patient_link']=='1S87UCuq']
allEnroll.columns#['Hospital_Admit_Date']
allEnroll.sample(10)

labs=pd.read_csv('Cardiac Program_M - patient labs.csv')
labs.columns
labs.groupby('patient_link').size()
labs.shape #(4148, 29)
labs_arc=pd.read_csv('Cardiac Program_Archive - patient labs.csv')
allLabs.loc[allLabs['patient_link']=='1S87UCuq'].groupby('labs_date_x').size()
allLabs=pd.merge(labs,labs_arc,on='patient_link',how='outer')
allLabs.groupby('patient_link').size()

#cp=Cardiac Program
cp=pd.read_excel('Cardiac Program_M.xlsx',sheet_name=None)
cp['patients']
cp.keys()
