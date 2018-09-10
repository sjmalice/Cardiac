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
    newKey['enrollId'] = newKey.patient_link + '#' + newKey.visitNo.map(str)
    return(newKey)
