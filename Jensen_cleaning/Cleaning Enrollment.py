
# coding: utf-8

# In[136]:


import pandas as pd
import numpy as np
import re


# In[137]:


df_enroll=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name='patient_enrollment_records')
#pd.read_csv('Cardiac Program_Archive - patient_enrollment_records.csv')


# In[138]:


#print('original shape', df_enroll.shape)
#df_enroll=df_enroll[df_enroll['create_user']!='multitechvisions@gmail.com']
#print('shape after removing multitechvision rows', df_enroll.shape)


# In[139]:


#Cleaning AICD
df_enroll.AICD=df_enroll.AICD.str.lower()
df_enroll=df_enroll.replace({'AICD':{'none':0, 'no':0, 'no aicd or pacemaker':0, '0' : 0,'0.25':0, '25%':0, 'o':0, '9/13/2017' : 0, 'lisinopril':0}})
df_enroll.AICD=df_enroll.AICD.fillna(9999) #to be imputed
allowed_vals=['0', 0, 9999]
df_enroll.loc[~df_enroll["AICD"].isin(allowed_vals), "AICD"] = 1
print(df_enroll['AICD'].value_counts())


# In[140]:


#Cleaning Acute_or_chronic
df_enroll.Acute_or_chronic=df_enroll.Acute_or_chronic.fillna(9999) #to be imputed
df_enroll.Acute_or_chronic.value_counts()


# In[141]:


#Cleaning Diagnosis_1

def lower_errors(x):
    try:
        return x.lower()
    except:
        return ""

def find_unique_diag(df_diag_column):
    """
    Within text Diagnosis Columns, returns a list of the Unique Diagnoses,
    removing the combinations of diagnoses
    """
    all_diag=df_diag_column.apply(lambda x: lower_errors(x)).unique()
    all_diag[7].split(' , ')
    unique_diag=[]
    for diag in all_diag:
        if len(diag)==0:
            continue
        else:
            unique_diag.append(diag.split(' , '))
    flat_list = [item for sublist in unique_diag for item in sublist]
    unique_diag=pd.Series(flat_list).unique()
    return unique_diag


# In[142]:


test=find_unique_diag(df_enroll.Diagnosis_1)


# In[143]:


df_enroll[['patient_link','Diagnosis_1']].sample(10)


# In[144]:



def dummify_diagnoses(df,unique_diag,diagnosis_col='Diagnosis_1'):
    """
    Takes Diagnoses and dummifies them for patients. If a patient has multiple
    diagnoses, will put a 1 in all relevant Diagnoses.
    The kth column is NA, no diagnosis. Maybe we will impute with the mode?
    """
    header=unique_diag.tolist().append('patient_link')
    dummy_diag=pd.DataFrame(columns=header)

    for row in range(df.shape[0]):
        pat_diag=lower_errors(df.iloc[row][diagnosis_col]).split(' , ')
        #print(pat_diag)
        dict_dummy_diag=dict(zip(unique_diag,np.zeros(len(unique_diag))))
        # dict_dummy_diag['patient_link']=df.iloc[row]['patient_link']
        #pd.DataFrame(np.zeros(len(unique_diag)).reshape(-1),columns=unique_diag)
        for diag in pat_diag:
            if diag in unique_diag:
                dict_dummy_diag[diag]=1
            else:
                continue
        tmp_dummy_diag=pd.DataFrame(dict_dummy_diag, index=[row])
        tmp_dummy_diag['patient_link']=df.iloc[row]['patient_link']
        dummy_diag = pd.concat([dummy_diag,tmp_dummy_diag], axis=0)

    return dummy_diag


# In[145]:

dum_diag=dummify_diagnoses(df_enroll,test)

dum_diag[dum_diag['patient_link']=='fuybpFQY']


# In[146]:


pd.set_option('display.max_columns', 500)
df_enroll[df_enroll['patient_link']=='fuybpFQY']


# In[147]:


df_enroll_diag=pd.merge(df_enroll, dum_diag, left_index=True, right_index=True)


# In[148]:


df_enroll_diag


# In[149]:


df_enroll_diag.loc[df_enroll_diag['patient_link_x']=='fuybpFQY']


# In[150]:


# df_enroll_diag.to_csv('clean_enrollment.csv')
