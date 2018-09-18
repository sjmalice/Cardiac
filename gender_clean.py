import pandas as pd
import numpy as np
import gender_guesser.detector as gender
import re



def impute_gender(gender, name):
    """ This function takes any row that has missing values in the gender column and take their name and then runs get gender funciton on it
        It first cleans the column to make it usable for the function and then the runction is run
        to apply this use df.apply(lambda row: impute_gender(row['patient_gender'],row['name']),axis=1)
        This just imputes what the funciton does so you need another function to make all the genders say the same thing

        This is mutating to the gender column
    """
    try:
        if (type(gender)==str and len(gender)==0) or (type(gender)==float and np.isnan(gender)):
            name=name.replace(r'(?<=[.,])(?=[^\s])', r' ')
            name=name.replace('C,', '')
            name=name.lower().split()[1]
            name=name.capitalize()
            d = gender.Detector()
            return(d.get_gender(name))
        else:
            return gender
    except:
        return gender

def normalizing_gender(gender):
    """After the gender function sometimes female is spelled with a lowercase or says mostly_female so this
        funciton just normalizes everything
        Right now NaN's and unknown are being imputed as Male but this could be changed it desired
        Use this function with df.apply(lambda row: normalizing_gender(row['patient_gender']),axis=1)
        This is mutating
    """
    if (gender=='female') or (gender=='Female') or (gender=='mostly_female'):
        return 'Female'
    else:
        return 'Male'
