
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[309]:


df_meds=pd.read_excel('Data/Cardiac Program_Archive.xlsx',sheet_name='Cardiac_Meds')#pd.read_csv('Cardiac Program_Archive - Cardiac_Meds.csv')


# In[310]:


print('original shape', df_meds.shape)
df_meds=df_meds.loc[df_meds['create_user']!='multitechvisions@gmail.com']#delete rows with this user
print('shape after removing multitechvision rows', df_meds.shape)


# In[290]:


#Cleaning ACE
df_meds['ACE']=df_meds['ACE'].str.lower()
df_meds.loc[df_meds['ACE'].isin(['none', '0']), "ACE"]=0
df_meds=df_meds.loc[~df_meds['ACE'].str.contains('allergic', na=False)]
df_meds=df_meds.loc[df_meds['ACE']!='asdf']#removed test created by multitechvision
df_meds=df_meds.loc[df_meds['ACE']!='5'] #removed test
df_meds['ACE']=df_meds['ACE'].fillna(9999)
allowed_vals2=[0, 9999]
df_meds.loc[~df_meds["ACE"].isin(allowed_vals2), "ACE"] = 1
print(df_meds['ACE'].value_counts())
df_meds.head(10)

# In[262]:


#Cleaning BB
df_meds['BB']=df_meds['BB'].str.lower()
df_meds['BB']=df_meds['BB'].fillna(9999)
df_meds.loc[df_meds['BB'].isin(['none', '0']), "BB"]=0
df_meds.loc[~df_meds["BB"].isin(allowed_vals2), "BB"] = 1
df_meds['BB'].value_counts()


# In[263]:


#Cleaning Diuretics
df_meds['Diuretics']=df_meds['Diuretics'].str.lower()
df_meds['Diuretics']=df_meds['Diuretics'].fillna(9999)
df_meds.loc[df_meds['Diuretics'].isin(['none', '0']), "Diuretics"]=0
df_meds.loc[~df_meds["Diuretics"].isin(allowed_vals2), "Diuretics"] = 1
df_meds['Diuretics'].value_counts()


# In[264]:


#Cleaning Anticoagulant
df_meds['Anticoagulant']=df_meds['Anticoagulant'].str.lower()
df_meds['Anticoagulant']=df_meds['Anticoagulant'].fillna(9999)
df_meds.loc[df_meds['Anticoagulant'].isin(['none', '0']), "Anticoagulant"]=0
none_values=df_meds.loc[df_meds['Anticoagulant'].str.contains('none', na=False)]['Anticoagulant'].tolist()
df_meds.loc[df_meds['Anticoagulant'].isin(none_values),"Anticoagulant"]=0
df_meds.loc[~df_meds["Anticoagulant"].isin(allowed_vals2), "Anticoagulant"] = 1
df_meds['Anticoagulant'].value_counts()


# In[265]:


#Cleaning Ionotropes
df_meds['Ionotropes']=df_meds['Ionotropes'].str.lower()
df_meds['Ionotropes']=df_meds['Ionotropes'].fillna(9999)
df_meds.loc[df_meds['Ionotropes'].isin(['none', '0']), "Ionotropes"]=0
df_meds.loc[~df_meds["Ionotropes"].isin(allowed_vals2), "Ionotropes"] = 1
df_meds.Ionotropes.value_counts()


# In[280]:

df_meds.sample(10)
# df_meds.to_csv('clean_cardiac_meds.csv')
