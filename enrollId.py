import pandas as pd

def generateEnrollId(df, columns = ['patient_link', 'enrollment_date']):
    '''
    Takes patient enrollment records and generates new enrollId and count
    feature for each time the patient is enrolled

    Keyword Arguments
    -----------------
    df -- patient_enrollment_records as pandas df
    columns -- patient_link and enrollment_date columns as list. Only needed if
    column names have not had case lowered.
    '''
    # group by patient_link and Enrollment_Date in patient_enrollment_records
    # sheet to generate enroll_id for each cycle of patient through facility
    newKey = pd.DataFrame([key for key, _ in df.groupby(columns)],
    columns=columns)
    # add visit number feature
    visitNo = [1]
    for i in range(1, len(newKey)):
        if newKey.patient_link[i] == newKey.patient_link[i-1]:
            visitNo.append(visitNo[-1]+1)
        else:
            visitNo.append(1)
    newKey['visitNo'] = visitNo
    #add new enrollId feature that's unique for each new visit
    newKey['enrollId'] =\
    newKey.patient_link.map(str) + '_' + newKey.visitNo.map(str)
    return(newKey)

def addEnrollId(df, date_col, rootDf):
    '''
    Adds new enrollId column to any data frame, df, containing a 'patient_link'
    and date column, date_col, using enrollId data frame generated by
    "generateEnrollId" function.

    Keyword Arguments
    -----------------
    df -- The date frame that new enrollId is added to.
    date_col -- the column of data frame df as string that contains the date to
    compare to date in rootDf.
    rootDf -- A data frame assumed to be generated by the "generateEnrollId"
    function.
    '''
    #add new enrollID column
    df['enrollId'] = pd.Series(index = df.index)
    # add enrollID for each row in rootDf when patient_link match and date is
    # greater than enrollment_date. Earlier enrollment_date are checked first
    # and overwritten when later dates are checked after
    for i in rootDf.index:
        df.loc[(df.patient_link == rootDf.patient_link[i]) &\
        (df[date_col] >= rootDf.enrollment_date[i]), "enrollId"]\
        = rootDf.enrollId[i]
    return(df)
