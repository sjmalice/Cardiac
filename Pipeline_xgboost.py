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
# df[keep_cols].isnull().sum()
# df.drop(['admit_weight'],axis=1)#,inplace=True)

df=df[df['outcome']!=2]
# %%
# load the columns I want to keep while modelling
# df.columns
column_dict=read_pkl('Models/model_columns2.pkl')
keep_cols=column_dict['keep_cols']
pat_cols=column_dict['pat_cols']

final_imputation(df)

# we have 47 duration and 12 acute/chronic that we are currently dropping
# df[keep_cols].isnull().sum()
df=get_standardized_columns(df, standardize_cols= ['ef','bnp',
    'this_bnp_change', 'bun', 'cr', 'potasium',
    'this_cr_change', 'resting_hr', 'systolic', 'diastolic',
    'duration', 'age'])
###### any transformations will go here ########
log_cols=['ef', 'weight', 'bnp',
 'bun', 'cr','potasium']
df[log_cols].isnull().sum()

for col in log_cols:
    df[col]=np.log1p(df[col]+1)
    # plt.hist(train(df)[col].dropna())
    # plt.title(str(col))
    # plt.show()

# write_pkl(df,'model_data.pkl')

# %%

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score, precision_score, roc_curve

from sklearn.metrics import confusion_matrix, accuracy_score

# %%
x=train(df)[keep_cols].drop('outcome',axis=1).dropna()
y=train(df)[keep_cols].dropna()['outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=43)

param_grid={'learning_rate':np.linspace(0.01,0.99,20),'max_depth':np.arange(3,10)}
model = XGBClassifier() #
model=GridSearchCV(estimator=model, param_grid= param_grid, scoring='precision', cv=5, return_train_score=True) #param_grid=grid_param,
model.fit(x_train,y_train)
model.cv_results_
cross_val_score(model,x_train,y_train,cv=5)

dir(model)
print(model.best_score_)
print(model.best_params_)
# model.fit(x_train, y_train)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# print(para_search.best_score_)
# print(para_search.best_params_)
# prediction=para_search.predict(x_test)
precision_score(y_test,predictions)
cnf_matrix = confusion_matrix(y_test, predictions)

prediction=model.predict_proba(x_test)
# prediction
low_threshold_pred=pd.Series(prediction[:,1]).apply(lambda x: 0 if x<0.75 else 1)

precision_score(y_test,low_threshold_pred)
cnf_matrix = confusion_matrix(y_test, low_threshold_pred)
cnf_matrix

accuracy = accuracy_score(y_test, low_threshold_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

## I want something like this, maybe a dataframe with predictions, probabilites
predictions=para_search.predict(test(df)[keep_cols].drop('outcome',axis=1).dropna())
# roc_curve(y_test,prediction[:,1])
