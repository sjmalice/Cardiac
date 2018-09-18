import pandas as pd
import numpy as np
from Clean_Fun import *
from Meta_fun import *
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
# %% Load dataset
df=pd.read_csv('Data/after_merge.csv',index_col=0)
# %% test patients, determing Response Value
pd.set_option('display.max_columns', 60)

df=meta_clean(df)
df[keep_cols].isnull().sum()

df=df[df['outcome']!=2]
# %%
# load the columns I want to keep while modelling
df.columns
column_dict=read_pkl('Models/model_columns.pkl')
keep_cols=column_dict['keep_cols']
pat_cols=column_dict['pat_cols']

final_imputation(df)

# we have 47 duration and 12 acute/chronic that we are currently dropping
df[keep_cols].isnull().sum()

###### any transformations will go here ########
log_cols=['ef','admit_weight', 'weight', 'bnp',
 'bun', 'cr', 'potasium']
df[log_cols].isnull().sum()

for col in log_cols:
    df[col]=np.log1p(df[col]+1)
    plt.hist(train(df)[col].dropna())
    plt.title(str(col))
    plt.show()
# %%

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools

# %%
x=train(df)[keep_cols].drop('outcome',axis=1).dropna()
y=train(df)[keep_cols].dropna()['outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y)

logistic = linear_model.LogisticRegression()
grid_param=10**np.linspace(-2,5,50)
my_param_grid = {'C': grid_param }
para_search = GridSearchCV(estimator=logistic, param_grid= my_param_grid, scoring='accuracy', cv=5, return_train_score=True) #param_grid=grid_param,
para_search.fit(x_train,y_train)
print(para_search.best_score_)
print(para_search.best_params_)
prediction=para_search.predict(x_test)
cnf_matrix = confusion_matrix(y_test, prediction)
cnf_matrix
# write_pkl(para_search,'Models/log_regression_notrans.pkl')
predictions=para_search.predict(test(df)[keep_cols].drop('outcome',axis=1).dropna())
