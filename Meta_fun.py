import pandas as pd
import numpy as np
from Clean_Fun import *
from Meta_fun import *

def train(df):
    """
    returns only the Train dataset, which you can act upon without taking it
    out of the whole dataset
    """
    return df[~df['outcome'].isnull()]
def test(df):
    """
    returns only the Test dataset, which you can act upon without taking it
    out of the whole dataset
    """
    return df[df['outcome'].isnull()]

def final_imputation(df):
    """
    Imputation so that I can practice a model with the API
    One thing to note is the use of train(df) to calculate the average or mode,
    but acting on the entire dataset (including test)
    Mutating Function
    """
    from scipy.stats import mode

    df.patient_gender.fillna(mode(train(df).patient_gender).mode[0],inplace=True)
    df.ef.fillna(np.nanmedian(train(df).ef),inplace=True)

    # maybe drop weight:
    # df.admit_weight.fillna(WEIGHT),inplace=True)

    # impute weight_change_since_admit using admit_weight
    df.weight_change_since_admit.fillna(np.nanmedian(train(df).weight_change_since_admit),inplace=True)
    df.this_weight_change.fillna(np.nanmedian(train(df).this_weight_change),inplace=True)# fill with median

    df.this_bnp_change.fillna(np.nanmedian(train(df).this_bnp_change),inplace=True)
    df.bnp.fillna(np.nanmedian(train(df).bnp),inplace=True)

    df.bun.fillna(np.nanmedian(train(df).bun),inplace=True)
    df.cr.fillna(np.nanmedian(train(df).cr),inplace=True)

    df.potasium.fillna(np.nanmedian(train(df).potasium),inplace=True)

    df.this_cr_change.fillna(np.nanmedian(train(df).this_cr_change),inplace=True)
    df.resting_hr.fillna(np.nanmedian(train(df).resting_hr),inplace=True)


    df.diastolic.fillna(np.nanmedian(train(df).diastolic),inplace=True)
    df.systolic.fillna(np.nanmedian(train(df).systolic),inplace=True)

    df.age.fillna(np.mean(train(df).age),inplace=True)
    print("Successfully imputed for all missing values")


def temporary_imputation(df):
    """
    Very temporary imputation so that I can practice a model with the API
    One thing to note is the use of train(df) to calculate the average or mode,
    but acting on the entire dataset (including test)
    Mutating Function
    """
    from scipy.stats import mode

    df.patient_gender.fillna(mode(train(df).patient_gender).mode[0],inplace=True)
    df.ef.fillna(np.mean(train(df).ef),inplace=True)
    # Maybe should group by Gender to impute weight
    df.admit_weight.fillna(np.mean(train(df).admit_weight),inplace=True)
    df.weight_change_since_admit.fillna(0,inplace=True)
    df.this_bnp_change.fillna(mode(train(df).this_bnp_change).mode[0],inplace=True)
    df.potasium.fillna(np.mean(train(df).potasium),inplace=True)
    df.this_cr_change.fillna(mode(train(df).this_cr_change).mode[0],inplace=True)
    df.resting_hr.fillna(np.mean(train(df).resting_hr),inplace=True)
    df.diastolic.fillna(np.mean(train(df).diastolic),inplace=True)
    df.systolic.fillna(np.mean(train(df).systolic),inplace=True)
    df.age.fillna(np.mean(train(df).age),inplace=True)
    print("Successfully imputed for all missing values")

def meta_clean(df):
    """
    All of our cleaning functions called at once
    Does not drop any columns
    Calculates outcome but returns entire dataset
    I added 2 lines to dummify Facility Link
    """
    # %% Clean effusion rate

    df['ef']=df['ef'].apply(lambda x: clean_EF_rows(x))
    df.ef[df.ef > 1] = np.nan

    # Remove oulier in this_bnp_change
    df.this_bnp_change[df.this_bnp_change < -5000] = np.nan

    # Clean Blood Pressure rows
    df['diastolic']=df.apply(lambda row: clean_diastolic_columns(row['diastolic'],
                        row['resting_bp'],'di',row['systolic']),axis=1)
    df['systolic']=df.apply(lambda row: clean_diastolic_columns(
        row['systolic'],row['resting_bp'],'sys',row['diastolic']),axis=1)

    # Dummify the diagnoses
    uniq_diag=find_unique_diag(df.diagnosis_1)
    dummy_df_diag=dummify_diagnoses(df,uniq_diag,diagnosis_col='diagnosis_1')
    df.drop('diagnosis_1',axis=1,inplace=True)
    dummy_df_diag.columns=pd.Series(uniq_diag).apply(lambda x: remove_paren(x)).append(pd.Series('enrollId'))
    df=df.merge(dummy_df_diag,on='enrollId',how="inner")

    # clean HR
    df['resting_hr']=df.resting_hr.apply(lambda x: hand_dates(x))

    # Clean Meds and aicd
    # acute or chronic
    med_aicd_clean(df,'ace', 0)
    med_aicd_clean(df,'bb', 0)
    med_aicd_clean(df,'diuretics', 0)
    med_aicd_clean(df,'anticoagulant', 0)
    med_aicd_clean(df,'ionotropes', 0)
    med_aicd_clean(df,'aicd', 0)
    med_aicd_clean(df,'other cardiac meds',0)

    df['enrollment_date']=df.enrollment_date.apply(lambda x: pd.to_datetime(x))
    df['discharge_date']=df.discharge_date.apply(lambda x: pd.to_datetime(x))
    df['date_of_birth']=df.date_of_birth.apply(lambda x: pd.to_datetime(x))

    # weight_dur_age_clean(df,dur_na=9999,age_na=9999,weight_perc_cutoff=0.2)
    df['duration']=df.apply(lambda row: find_duration(row['discharge'],
        row['enrollment_date'],row['discharge_date']),axis=1)
    df['age'] = df['date_of_birth'].apply(find_age)
    df['weight_change_since_admit'] = df.apply(lambda row: get_frac_weight_change(
        row['weight'],row['weight_change_since_admit']),axis=1)
    df['this_weight_change'] = df.apply(lambda row: get_frac_weight_change(
        row['weight'],row['this_weight_change']),axis=1)
    df['weight_change_since_admit_frac'] = df.apply(lambda row: get_frac_weight_change(row['weight'],row['weight_change_since_admit']),axis=1)
    df['this_weight_change_frac'] = df.apply(lambda row: get_frac_weight_change(row['weight'],row['this_weight_change']),axis=1)
    df.drop('this_weight_change', axis =1,inplace =True)
    df.drop('weight_change_since_admit', axis =1,inplace =True)
    df.drop('admit_weight', axis =1,inplace =True)

    df['patient_gender']=df.patient_gender.apply(lambda x: clean_gender(x))
    df['acute_or_chronic']=df.apply(lambda row: impute_acute_chronic(row['acute_or_chronic'],row['duration']),axis=1)

    # set any 0 lab results to None
    labs=['bnp','cr','potasium','mg','sodium']
    for lab in labs:
        df[lab] = df[lab].apply(lambda x: clean_labs(x))

    remove_invalid_rows(df)

    df.duration=df.duration.apply(lambda x: None if x==9999 else x)
    df.age=df.age.apply(lambda x: None if x==9999 else x)

    df['outcome']=df.apply(lambda row: determine_outcome(row['status'],row['discharge'],row['discharge_date']),axis=1)

    # %%
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.columns = [x.replace(" ", "_") for x in df.columns]
    df = drop_date_cols(df)

    dummies=pd.get_dummies(df['facilities_link'],prefix='F')
    df = pd.concat([df.drop("facilities_link",axis=1), dummies], axis=1)

    return df
