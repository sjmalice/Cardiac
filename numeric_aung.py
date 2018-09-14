%load_ext autoreload
%autoreload 2
from data_merge import *
from Clean_Fun import *
import re
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display
pd.options.display.max_columns = None #show all columns
import datetime

# %%

live_path='Data/Cardiac Program_M.xlsx'
archive_path='Data/Cardiac Program_Archive.xlsx'
live_sheet_pkl='pickle_jar/live_sheets.pkl'
archive_sheet_pkl='pickle_jar/archive_sheets.pkl'
datecol_pkl='pickle_jar/datecols.pkl'
#df = sheet_merge(live_path, archive_path,
    live_sheet_pkl, archive_sheet_pkl, datecol_pkl)
# %%
#df.to_csv('sheet_merge.csv')


df.shape
df.columns

df_new = pd.read_csv('sheet_merge.csv')
df_new.drop(df_new.columns[0], axis =1,inplace=True)
df_new.shape
df_new['enrollment_date'] = pd.to_datetime(df_new['enrollment_date'])
df_new['discharge_date'] = pd.to_datetime(df_new['discharge_date'])
df_new['date_of_birth'] = pd.to_datetime(df_new['date_of_birth'])
df_new[['discharge','discharge_date','enrollment_date','date_of_birth','weight','weight_change_since_admit']].head()

today = datetime.datetime.today()
today
df.discharge_date.apply(lambda x: (today-x)/np.timedelta64(1, 'D'))

def find_duration(discharge, enroll_date, discharge_date):
    """
    duration (type float, expressed in days) is difference of discharge_date and
    admission_date if discharge is true, otherwise duration is time since admission_dateself.
    Non mutating
    Use like: df['duration']=df.apply(lambda row: find_duration(row['discharge'],
        row['enrollment_date'],row['discharge_date']),axis=1)
    """
    #pass
    today = datetime.datetime.today()
    if discharge : #True
        return (discharge_date - enroll_date).days
    else:
        return (today - enroll_date).days

df['duration']=df.apply(lambda row: find_duration(row['discharge'],row['enrollment_date'],row['discharge_date']),axis=1)
df.duration.head()


def find_age(row):
    """
    age (type float, expressed in years) is deduced from birth date
    Non mutating
    Use as df['age'] = df['date_of_birth'].apply(find_age)
    """
    #pass
    today = datetime.datetime.today()
    try:
        x = round((today - row).days/365)
    except ValueError:
        x = np.nan
    return x
df['age'] = df['date_of_birth'].apply(find_age)
df.age.head()

df[['weight','weight_change_since_admit']].head()

def clean_weight_change(weight, weight_change):
    """
    If abs(weight_change)/ weight > 0.2:
    weight_change (float, expressed in pounds) is recursively divided by 10 until abs(weight_change)/ weight < 0.2
    Non Mutating
    Use like df['weight_change_since_admit'] = df.apply(lambda row: clean_weight_change(row['weight'],row['weight_change_since_admit']),axis=1)
    """
    pass
    if abs(weight_change)/weight < 0.2:
        return weight_change
    else:
        while abs(weight_change)/weight > 0.2:
            weight_change /= 10
        return weight_change
    #df['weight_change_since_admit'] = np.where(abs(df['weight_change_since_admit']/ \
    #    df['weight']) < 0.2, df['weight_change_since_admit'], df['weight_change_since_admit']/10)

df['weight_change_since_admit']=df.apply(lambda row: clean_weight_change(row['weight'],row['weight_change_since_admit']),axis=1)
df['this_weight_change2']=df.apply(lambda row: clean_weight_change(row['weight'],row['this_weight_change']),axis=1)
df['weight_change_since_admit2'].hist()
df['weight_change_since_admit'].hist()
df['this_weight_change2'].hist()
df['this_weight_change'].hist()

df[['weight','weight_change_since_admit', 'weight_change_since_admit2']].head(10)

df['this_weight_change']=df.apply(lambda row: clean_weight_change(row['weight'],
        row['this_weight_change']),axis=1)

def clean_numerical(df):
    """
    This function adds duration and age columns and naively
    cleans the weight_change_since_admit.
    duration (type int, expressed in days) is difference of discharge_date and
    admission_date if discharge is true, otherwise duration is time since admission_dateself.
    age (type int, expressed in years) is deduced from birth date

    If abs(weight_change_since_admit)/ weight < 0.2:
    weight_change_since_admit (float, expressed in pounds) is divided by 10
    otherwise it is unchanged.
    """
    ## duration
    today = pd.to_datetime('today')
    df.loc[df['discharge']==True,'duration'] = (pd.to_datetime(df.loc[df['discharge']==True, \
        'discharge_date']) - pd.to_datetime(df.loc[df['discharge']==True,'enrollment_date']))
    df.loc[df['discharge']==False,'duration'] = (today - \
        pd.to_datetime(df.loc[df['discharge']==False,'enrollment_date']))
    df.duration = df.duration.fillna(-9999999)
    df['duration'] = (df['duration'] / np.timedelta64(1, 'D')).astype(int)

    ##age
    df['age'] = (today - pd.to_datetime(df['date_of_birth'])).apply(lambda x: \
        float(x.days)/365).fillna(-99.).astype(int)
    ## weight_change_since_admit
    df['weight_change_since_admit'] = np.where(abs(df['weight_change_since_admit']/ \
        df['weight']) < 0.2, df['weight_change_since_admit'], df['weight_change_since_admit']/10)

    return

clean_numerical(df)
df.shape
df.duration.sample(5)
df.age.sample(5)
df.weight_change_since_admit.isnull().value_counts()
