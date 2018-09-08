import pandas as pd


cpa=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name=['patient_enrollment_records'])
cp=pd.read_excel('Data/Cardiac Program_M.xlsx',sheet_name=['patients'])

dfa=cpa['patient_enrollment_records']
df=cp['patients']
archivepatients=dfa.patient_link.unique()
patients=df.patient_id

missing_pats=[]
for pat in archivepatients:
    if pat in patients.values:
        print(pat)
        print('-'*50)
        continue
    else:
        # print(pat)
        missing_pats.append(pat)
len(missing_pats)
len(archivepatients)
