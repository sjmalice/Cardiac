from data_merge import *
from Clean_Fun import *
import re
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn import preprocessing

df = pd.read_csv('pipeline_cleaned.csv')
df.drop(df.columns[0], axis =1,inplace=True)  #first column was index from csv
df.shape
df.sample(5)

df.isnull().sum()
#df.isnull().sum().sort_values().plot.barh()
df.isnull().sum().sort_values(ascending = False).plot.bar()
plt.tight_layout()
#plt.savefig('null.png')


numerics = ['int64',  'float64']
df_numeric = df.select_dtypes(include=numerics)
numeric_columns = df_numeric.columns
df_numeric.apply(np.min).sort_values().plot.bar()
df_numeric.apply(np.max).sort_values().plot.bar()
#df_numeric = df_numeric.drop('duration', axis =1)
#df_numeric = df_numeric.drop('bnp', axis =1)
df_numeric = df_numeric.fillna(-1000)

x = df_numeric.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df_numeric = pd.DataFrame(x_scaled)
df_numeric.columns = numeric_columns
#df_numeric.head()

df2 = pd.concat([df_numeric, df['acute_or_chronic']], axis=1)
#df2.drop('acute_or_chronic', axis =1, inplace = True)
df2['acute_or_chronic']

#Special status has many null values
df.special_status.value_counts()
## Things to look at
# Acute vs Chronic
# All Diagnoses
# Medicines
# Pacemaker (AICD)
len(df.patient_link.unique())

df.date_of_birth = pd.to_datetime(df.date_of_birth)
df.date_of_birth.hist()

today = pd.to_datetime('today')
df[df.date_of_birth>today]['age']

df.groupby('patient_link').count()['date_of_birth'].sort_values(ascending = False).head(10) #NUmber of entries

df[df['patient_link']=='s2SbAKQm']


#Acute vs Chronic
df['acute_or_chronic'].isnull().value_counts()
df['acute_or_chronic'].value_counts()
#df[df['acute_or_chronic'].isnull()]['acute_or_chronic'].head()


pd.DataFrame(df.groupby('acute_or_chronic').mean())[['ace','bb', 'diuretics', 'anticoagulant','ionotropes']]
#On average what kind of meds they were on?
acuteChronic_med = pd.DataFrame(df.groupby('acute_or_chronic').mean())[['ace','bb', 'diuretics', 'anticoagulant','ionotropes']]
acuteChronic_med.plot(kind='bar', stacked =True)

acuteChronic_med.transpose().plot.barh()

#acuteChronic_gender =
(df.groupby(['acute_or_chronic','patient_gender']).count())['patient_link'].plot.barh()

df.groupby('acute_or_chronic').mean()[['weight','weight_change_since_admit']]
df.groupby('acute_or_chronic').mean()[['weight_change_since_admit']].transpose().plot.barh()
df.groupby('acute_or_chronic').mean()[['cr']].transpose().plot.barh()
df.groupby('acute_or_chronic').mean()[['this_cr_change']].transpose().plot.barh()
df.groupby('acute_or_chronic').mean()[['bnp']].transpose().plot.barh()

df.groupby('acute_or_chronic').mean()[['atrial fibrilation','cad/mi','cardiomyoapthy','heart failure unspecfied', 'lvad','systolic chf (ef<60%)','diastolic heart failure (ef >50%)']].transpose().plot.barh()

# Labs: 'cr', 'this_cr_change','sodium', 'potasium', 'mg',
##'atrial fibrilation', 'cad/mi', 'cardiomyoapthy',
##'diastolic heart failure (ef >50%)', 'heart failure unspecfied', 'lvad','systolic chf (ef<60%)',
df[['atrial fibrilation','cad/mi','cardiomyoapthy','diastolic heart failure (ef >50%)', 'heart failure unspecfied', 'lvad','systolic chf (ef<60%)']].hist()

df.lvad.value_counts()
df[df.lvad==1]['outcome']


df[['bun','sodium','potasium','mg']].hist()


df[['cr','this_cr_change']].hist()

#date of birth
df.date_of_birth.hist()

df.loc[df.date_of_birth < today,'date_of_birth'].hist()
len(df[df.date_of_birth < today])
len(df[df.date_of_birth >= today])

# BNP 'bnp_date', 'bnp', 'this_bnp_change'

df.bnp_date = pd.to_datetime(df.bnp_date)
df.bnp_date.hist()
df.loc[df.bnp_date<today,'bnp_date'].hist() #identical
len(df[df.bnp_date < today])
len(df[df.bnp_date >= today])


df.bnp.max(), df.bnp.min()
df.bnp.hist()
df[df.bnp<1000]['bnp'].hist()
df[df.bnp<100000]['bnp'].map(np.log).hist()
len(df[df.bnp>1000])

df.this_bnp_change.min(), df.this_bnp_change.max()
df.this_bnp_change.hist()
len(df[df.this_bnp_change< -1000])
df[df.this_bnp_change< -1000]['bnp'].hist()
df['this_bnp_change'].hist()
# BP: 'bp_date', 'systolic', 'diastolic', 'resting_bp',
df.bp_date = pd.to_datetime(df.bp_date)
df.bp_date.hist() #problematic

df.loc[df.bp_date<today,'bp_date'].hist()
len(df[df.bp_date < today])
len(df[df.bp_date >= today])

df[['systolic','diastolic']].hist()
df['resting_bp'].apply(lambda x: str(x).split('/')[0]).head()
df['systolic'].head()


# Length of Stay (Duration): 'duration'
df.loc[df.duration> -1000,'duration'].hist()
df.loc[df.duration> -1000,'duration'].map(np.log).hist()
df.duration.shape
df[df.duration> -1000]['duration'].shape



df.dtypes


#Normalized, Not helpful for camparison ?
df2.groupby('acute_or_chronic').mean()[['weight','this_weight_change','cr','this_cr_change','bnp']].transpose().plot.barh()

df2.groupby('acute_or_chronic').mean()[['atrial fibrilation','cad/mi','cardiomyoapthy','heart failure unspecfied', 'lvad','systolic chf (ef<60%)','diastolic heart failure (ef >50%)']].transpose().plot.barh()
df2.groupby('acute_or_chronic').mean()[['ef','admit_weight','bnp','this_bnp_change', 'systolic','diastolic']].transpose().plot.barh()

df2.groupby('acute_or_chronic').mean()[['ace','bb','diuretics','anticoagulant','ionotropes']].transpose().plot.barh()

df.this_weight_change.hist(bins =20)
df.weight_change_since_admit.hist(bins =20)

df[df.bb==0]['acute_or_chronic'].value_counts() #this gives hint to chronic/acute
df[(df.cardiomyoapthy==1) ]['acute_or_chronic'].value_counts()

corr_matrix = df_numeric.corr()
print(len(corr_matrix))
corr_matrix.head()
#corr_matrix.SalePrice.sort_values(ascending=False).drop(['SalePrice'])
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns.values,yticklabels=corr_matrix.columns.values)
#plt.tight_layout()
#plt.savefig('corr.png')
