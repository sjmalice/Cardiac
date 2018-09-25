import pandas as pd
import numpy as np
from Clean_Fun import *
from Meta_fun import *
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2
# %% Load dataset
df=pd.read_csv('SuccessArchivetoo2.csv',index_col=0)
# %% test patients, determing Response Value
pd.set_option('display.max_columns', 60)

df=meta_clean(df)
# df[keep_cols].isnull().sum()
# df.drop(['admit_weight'],axis=1)#,inplace=True)

df=df[df['outcome']!=2]
# %%
# load the columns I want to keep while modelling
df.columns
column_dict=read_pkl('Models/model_columns.pkl')
keep_cols=column_dict['keep_cols']
pat_cols=column_dict['pat_cols']

final_imputation(df)

# we have 47 duration and 12 acute/chronic that we are currently dropping
# df[keep_cols].isnull().sum()

###### any transformations will go here ########
log_cols=['ef', 'bnp',
 'bun', 'cr', 'potasium']
df[log_cols].isnull().sum()

for col in log_cols:
    df[col]=np.log1p(df[col]+1)
    # plt.hist(train(df)[col].dropna())
    # plt.title(str(col))
    # plt.show()

keep_cols=['patient_gender',
 'ef',
 'weight',
 'this_weight_change_frac',
 'weight_change_since_admit_frac',
 'bnp',
 'this_bnp_change',
 'ace',
 'bb',
 'diuretics',
 'anticoagulant',
 'ionotropes',
 'other_cardiac_meds',
 'bun',
 'cr',
 'potasium',
 'this_cr_change',
 'resting_hr',
 'systolic',
 'diastolic',
 'outcome',
 'cad/mi',
 'heart_failure_unspecfied',
 'diastolic_heart_failure',
 'systolic_chf',
 'atrial_fibrilation',
 'cardiomyoapthy',
 'lvad',
 # 'chf',
 'duration',
 'age',
 'F_5nKZ993n',
 'F_71ADiKaS',
 'F_Fy1r9IXM',
 'F_KYzNhByH',
 'F_L1V04aB0',
 'F_US4llDDz',
 'F_Xxk5Yn3E',
 'F_kIUZIzRp',
 'F_mB0G57bu']
# %%

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, roc_curve
import itertools

# %%
x=train(df)[keep_cols].drop('outcome',axis=1).dropna()
y=train(df)[keep_cols].dropna()['outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=43)

model = linear_model.SGDClassifier(loss='log',penalty='l2')
# grid_param=10**np.linspace(-5,5,50)
# my_param_grid = {'alpha': grid_param }
# model = GridSearchCV(estimator=sgd, param_grid= my_param_grid, scoring='precision', cv=5, return_train_score=True) #param_grid=grid_param,
model.fit(x_train,y_train)
# model.param_grid
# dir(model)

# print("Grid search score: "+str(model.best_score_))
# print(model.best_params_)
predictions=model.predict(x_test)
precision=precision_score(y_test,predictions)
print("Precision: "+str(precision))
cnf_matrix = confusion_matrix(y_test, predictions)
cnf_matrix
prediction=model.predict_proba(x_test)
# low_threshold_pred=pd.Series(prediction[:,1]).apply(lambda x: 0 if x<0.750 else 1)
# precision_score(y_test,low_threshold_pred)
# cnf_matrix = confusion_matrix(y_test, low_threshold_pred)
# cnf_matrix
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# write_pkl(model,'Models/log_regression_notrans.pkl')

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob):
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label =True)
    sum_sensitivity_specificity_train = tpr_train + (1-fpr_train) #this is the TNR
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]
    best_fpr_train = fpr_train[best_threshold_id_train]
    best_tpr_train = tpr_train[best_threshold_id_train]
    y_train = y_train_prob > best_threshold

    cm_train = confusion_matrix(y_train_true, y_train)
    acc_train = accuracy_score(y_train_true, y_train)
    auc_train = roc_auc_score(y_train_true, y_train_prob)

    print('Train Accuracy: %s ' %acc_train)
    print('Train AUC: %s ' %auc_train)
    print('Train Confusion Matrix:')
    print(cm_train)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    curve1 = ax.plot(fpr_train, tpr_train)
    curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    ax.text(best_fpr_train, best_tpr_train, s = '(%.3f,%.3f)' %(best_fpr_train, best_tpr_train))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (Train), AUC = %.4f'%auc_train)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_true, y_test_prob, pos_label =True)

    y_test = y_test_prob > best_threshold

    cm_test = confusion_matrix(y_test_true, y_test)
    acc_test = accuracy_score(y_test_true, y_test)
    auc_test = roc_auc_score(y_test_true, y_test_prob)

    print('Test Accuracy: %s ' %acc_test)
    print('Test AUC: %s ' %auc_test)
    print('Test Confusion Matrix:')
    print(cm_test)

    tpr_score = float(cm_test[1][1])/(cm_test[1][1] + cm_test[1][0])
    fpr_score = float(cm_test[0][1])/(cm_test[0][0]+ cm_test[0][1])

    ax2 = fig.add_subplot(122)
    curve1 = ax2.plot(fpr_test, tpr_test)
    curve2 = ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    dot = ax2.plot(fpr_score, tpr_score, marker='o', color='black')
    ax2.text(fpr_score, tpr_score, s = '(%.3f,%.3f)' %(fpr_score, tpr_score))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (Test), AUC = %.4f'%auc_test)
    plt.savefig('ROC', dpi = 500)
    plt.show()

    return best_threshold

y_train_prob=model.predict_proba(x_train)
pred_proba=model.predict_proba(x_test)

best_threshold=plot_ROC(y_train, y_train_prob[:,1], y_test, pred_proba[:,1])
print('Best threshold: %s' %best_threshold)

pred_proba=model.predict_proba(x_test)
low_threshold_pred=pd.Series(pred_proba[:,1]).apply(lambda x: 0 if x<best_threshold else 1)

thresh_precision=precision_score(y_test,low_threshold_pred)
cnf_thresh_matrix = confusion_matrix(y_test, low_threshold_pred)
thresh_accuracy = accuracy_score(y_test, low_threshold_pred)
# thresh_recall = recall_score(y_test,low_threshold_pred)
print("Accuracy: %.2f%%" % (thresh_accuracy * 100.0))
print("Precision: %.2f%%" % (thresh_precision * 100.0))
# print("Recall: %.2f%%" % (thresh_recall * 100.0))
print(cnf_thresh_matrix)


# predictions=model.predict(test(df)[keep_cols].drop('outcome',axis=1).dropna())
