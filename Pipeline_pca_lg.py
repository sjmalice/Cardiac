import pandas as pd
import numpy as np
from Clean_Fun import *
from Meta_fun import *
from scipy.stats import mode

# %% Load dataset
df=pd.read_csv('Data/after_merge.csv',index_col=0)
# %% test patients, determing Response Value


df=meta_clean(df)
df=df[df['outcome']!=2]

# %%
# load the columns I want to keep while modelling
column_dict=read_pkl('Models/model_columns.pkl')
keep_cols=column_dict['keep_cols']
pat_cols=column_dict['pat_cols']

temporary_imputation(df)
# there remains 20 annoying patients with many missing values
log_cols=['ef','admit_weight', 'weight', 'bnp',
 'bun', 'cr', 'potasium', 'age']
for col in log_cols:
    df[col]=np.log1p(df[col]+1)
    plt.hist(train(df)[col].dropna())
    plt.title(str(col))
    plt.show()
###### any transformations will go here ########

# %%

from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# %%
x=train(df)[keep_cols].drop('outcome',axis=1).dropna()
y=train(df)[keep_cols].dropna()['outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y)

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

pca.fit(x)

# %% Plo the PCA Spectrum

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# %%
n_components=[1,2,3,5,10]

Cs=10**np.linspace(-2,5,50)
my_param_dict = {'pca__n_components':n_components,
                    'logistic__C':Cs}
estimator = GridSearchCV(pipe,
                        my_param_dict)
                            # scoring='accuracy', cv=5,
                            #  return_train_score=True) #param_grid=grid_param,
estimator.fit(x,y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

estimator.best_score_

prediction=estimator.predict(x_test)
cnf_matrix = confusion_matrix(y_test, prediction)
cnf_matrix
