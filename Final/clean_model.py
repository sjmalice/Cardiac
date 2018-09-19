# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, roc_curve
from sklearn import linear_model
from sklearn.metrics import confusion_matrix


from IPython.display import display
pd.options.display.max_columns = None

from data_merge import *
from Clean_Fun import *
from Meta_fun import *

# NOTE have to use remove_invalid_rows() inside ALex's function,
# before we remove patient name
# %% Load dataset

def xgboost_model(df_full,threshold=0.75):

    keep_cols=['patient_gender', 'ef', 'acute_or_chronic',
           'weight','this_weight_change_frac','weight_change_since_admit_frac', 'bnp',
           'this_bnp_change','ace', 'bb', 'diuretics',
           'anticoagulant', 'ionotropes', 'other_cardiac_meds', 'bun',
           'cr', 'potasium', 'this_cr_change',
           'resting_hr', 'systolic', 'diastolic', 'outcome',
           'cad/mi', 'heart_failure_unspecfied', 'diastolic_heart_failure',
           'systolic_chf', 'atrial_fibrilation', 'cardiomyoapthy', 'lvad',
            'duration', 'age' ,'F_5nKZ993n', 'F_71ADiKaS', 'F_Fy1r9IXM',
           'F_KYzNhByH', 'F_L1V04aB0', 'F_US4llDDz', 'F_Xxk5Yn3E', 'F_kIUZIzRp',
           'F_mB0G57bu'] #'chf',
    robust_cols=[]
    for col in keep_cols:
        if col in df_full.columns:
            robust_cols.append(col)

    df=df_full[keep_cols]
    final_imputation(df)

    #Remove patients with unknown outcome
    df = df[df.outcome != 2]

    #Train-test split

    x=train(df).drop('outcome',axis=1).dropna()
    y=train(df).dropna()['outcome']


    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

    param_grid={'learning_rate':np.linspace(0.01,0.99,20),'max_depth':np.arange(3,8)}
    model = XGBClassifier(random_state=123) #
    grid_model= GridSearchCV(estimator=model, param_grid= param_grid, scoring='precision', cv=5, return_train_score=True) #param_grid=grid_param,
    grid_model.fit(x_train,y_train)
    print("Gridsearch best score: {}".format(grid_model.best_score_))
    print("Gridsearch results: {}".format(grid_model.best_params_))
    y_pred = grid_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%%" % (precision * 100.0))
    cnf_matrix = confusion_matrix(y_test, predictions)
    print(cnf_matrix)

    pred_proba=grid_model.predict_proba(x_test)
    low_threshold_pred=pd.Series(pred_proba[:,1]).apply(lambda x: 0 if x<threshold else 1)

    thresh_precision=precision_score(y_test,low_threshold_pred)
    cnf_thresh_matrix = confusion_matrix(y_test, low_threshold_pred)
    thresh_accuracy = accuracy_score(y_test, low_threshold_pred)
    print("Accuracy: %.2f%%" % (thresh_accuracy * 100.0))
    print("Precision: %.2f%%" % (thresh_precision * 100.0))
    print(cnf_thresh_matrix)
    outcome_df = df[df.outcome.isnull()]
    outcome_X = outcome_df.drop('outcome', axis=1)
    outcome_pred = pd.Series(grid_model.predict(outcome_X))
    outcome_proba = pd.Series(grid_model.predict_proba(outcome_X)[:,1])

    outcome_df_full = df_full[df_full['outcome'].isnull()]
    outcome_df_full.reset_index(drop = True,inplace =True)
    outcome_df_full = outcome_df_full[['patient_link','enrollId']]
    return_df =  pd.concat([outcome_df_full,outcome_pred, outcome_proba], axis =1)

    print("returning order: return_df, accuracy, precision, cnf_matrix")
    return return_df, accuracy, precision, cnf_matrix, thresh_accuracy, thresh_precision, cnf_thresh_matrix

def logistic_model(df_full,threshold=0.75):

    keep_cols=['patient_gender', 'ef', 'acute_or_chronic',
           'weight','this_weight_change_frac','weight_change_since_admit_frac', 'bnp',
           'this_bnp_change','ace', 'bb', 'diuretics',
           'anticoagulant', 'ionotropes', 'other_cardiac_meds', 'bun',
           'cr', 'potasium', 'this_cr_change',
           'resting_hr', 'systolic', 'diastolic', 'outcome',
           'cad/mi', 'heart_failure_unspecfied', 'diastolic_heart_failure',
           'systolic_chf', 'atrial_fibrilation', 'cardiomyoapthy', 'lvad',
            'duration', 'age' ,'F_5nKZ993n', 'F_71ADiKaS', 'F_Fy1r9IXM',
           'F_KYzNhByH', 'F_L1V04aB0', 'F_US4llDDz', 'F_Xxk5Yn3E', 'F_kIUZIzRp',
           'F_mB0G57bu'] #'chf',
    robust_cols=[]
    for col in keep_cols:
        if col in df_full.columns:
            robust_cols.append(col)

    df=df_full[keep_cols]
    final_imputation(df)

    #Remove patients with unknown outcome
    df = df[df.outcome != 2]

    #Train-test split

    x=train(df).drop('outcome',axis=1).dropna()
    y=train(df).dropna()['outcome']


    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

    logistic = linear_model.LogisticRegression()
    grid_param=10**np.linspace(-2,5,50)
    my_param_grid = {'C': grid_param }
    grid_model = GridSearchCV(estimator=logistic, param_grid= my_param_grid, scoring='precision', cv=5, return_train_score=True) #param_grid=grid_param,
    grid_model.fit(x_train,y_train)

    print("Gridsearch best score: {}".format(grid_model.best_score_))
    print("Gridsearch results: {}".format(grid_model.best_params_))
    y_pred = grid_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%%" % (precision * 100.0))
    cnf_matrix = confusion_matrix(y_test, predictions)
    print(cnf_matrix)

    pred_proba=grid_model.predict_proba(x_test)
    low_threshold_pred=pd.Series(pred_proba[:,1]).apply(lambda x: 0 if x<threshold else 1)

    thresh_precision=precision_score(y_test,low_threshold_pred)
    cnf_thresh_matrix = confusion_matrix(y_test, low_threshold_pred)
    thresh_accuracy = accuracy_score(y_test, low_threshold_pred)
    print("Accuracy: %.2f%%" % (thresh_accuracy * 100.0))
    print("Precision: %.2f%%" % (thresh_precision * 100.0))
    print(cnf_thresh_matrix)
    outcome_df = df[df.outcome.isnull()]
    outcome_X = outcome_df.drop('outcome', axis=1)
    outcome_pred = pd.Series(grid_model.predict(outcome_X))
    outcome_proba = pd.Series(grid_model.predict_proba(outcome_X)[:,1])

    outcome_df_full = df_full[df_full['outcome'].isnull()]
    outcome_df_full.reset_index(drop = True,inplace =True)
    outcome_df_full = outcome_df_full[['patient_link','enrollId']]
    return_df =  pd.concat([outcome_df_full,outcome_pred, outcome_proba], axis =1)
    return_df.columns=['patient_link','enrollId','class_predict','Probability']
    print("returning order: return_df, accuracy, precision, cnf_matrix")
    return return_df, accuracy, precision, cnf_matrix, thresh_accuracy, thresh_precision, cnf_thresh_matrix
