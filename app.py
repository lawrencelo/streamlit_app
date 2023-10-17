import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import joblib
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Sequential

scaler = load('scaler.pkl')
nn_model = load_model("nn_model.h5")
knn = joblib.load("knn_model.pkl")
xgb = joblib.load("xgb_model.pkl")
rfc = joblib.load("rfc_model.pkl")

st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create two main columns: image_column and form_column
image_column, form_column = st.columns([1, 4])

from PIL import Image
image = Image.open('HALV.jpg')

#st.image(image,    use_column_width=True)

image_column.title("HALV Prediction")
image_column.image(image, use_column_width=True)

def set_employment_type_value(df, employment_type_value):
    # Check if the column exists in the DataFrame
    if employment_type_value in df.columns:
        df[employment_type_value] = 1
    return df


def predict_loan_default(UNIQUEID, DISBURSED_AMOUNT, ASSET_COST, LTV, BRANCH_ID, SUPPLIER_ID, 
        MANUFACTURER_ID, CURRENT_PINCODE_ID, DATE_OF_BIRTH, EMPLOYMENT_TYPE, 
        DISBURSAL_DATE, STATE_ID, EMPLOYEE_CODE_ID, MOBILENO_AVL_FLAG, AADHAR_FLAG, 
        PAN_FLAG, VOTERID_FLAG, DRIVING_FLAG, PASSPORT_FLAG, PERFORM_CNS_SCORE, 
        PERFORM_CNS_SCORE_DESCRIPTION, PRI_NO_OF_ACCTS, PRI_ACTIVE_ACCTS, 
        PRI_OVERDUE_ACCTS, PRI_CURRENT_BALANCE, PRI_SANCTIONED_AMOUNT, 
        PRI_DISBURSED_AMOUNT, SEC_NO_OF_ACCTS, SEC_ACTIVE_ACCTS, SEC_OVERDUE_ACCTS, 
        SEC_CURRENT_BALANCE, SEC_SANCTIONED_AMOUNT, SEC_DISBURSED_AMOUNT, 
        PRIMARY_INSTAL_AMT, SEC_INSTAL_AMT, NEW_ACCTS_IN_LAST_SIX_MONTHS, 
        DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS, AVERAGE_ACCT_AGE, 
        CREDIT_HISTORY_LENGTH, NO_OF_INQUIRIES, model_choice):
    # Create a one row dataframe with the provided column names and initial value of 0.0

    columns = [
        "AADHAR_FLAG", "VOTERID_FLAG", "NEW_ACCTS_IN_LAST_SIX_MONTHS", 
        "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", "PRI_OVERDUE_ACCTS", "NO_OF_INQUIRIES",
        "Salaried", "Self employed", "Unclassified", "A-Very Low Risk", "B-Very Low Risk",
        "C-Very Low Risk", "D-Very Low Risk", "E-Low Risk", "F-Low Risk", "G-Low Risk",
        "H-Medium Risk", "I-Medium Risk", "J-High Risk", "K-High Risk", "L-Very High Risk",
        "M-Very High Risk", "No Bureau History Available", "Not Scored: More than 50 active Accounts found",
        "Not Scored: No Activity seen on the customer (Inactive)", "Not Scored: No Updates available in last 36 months",
        "Not Scored: Not Enough Info available on the customer", "Not Scored: Only a Guarantor",
        "Not Scored: Sufficient History Not Available", "DISBURSED_AMOUNT", "ASSET_COST", "LTV",
        "PERFORM_CNS_SCORE", "PRI_CURRENT_BALANCE", "PRI_DISBURSED_AMOUNT", "SEC_CURRENT_BALANCE",
        "SEC_DISBURSED_AMOUNT", "PRIMARY_INSTAL_AMT", "SEC_INSTAL_AMT", "AVERAGE_ACCT_AGE",
        "CREDIT_HISTORY_LENGTH", "DISBURSAL_AGE"
    ]

    # Create a DataFrame with a single row of zeros
    initial_values = [0.0] * len(columns)
    single_row_df = pd.DataFrame([initial_values], columns=columns)
    
    single_row_df = set_employment_type_value(single_row_df.copy(), EMPLOYMENT_TYPE)

    single_row_df = set_employment_type_value(single_row_df.copy(), PERFORM_CNS_SCORE_DESCRIPTION)

    single_row_df['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'] = 0 if DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS == 0 else 1
    single_row_df['NEW_ACCTS_IN_LAST_SIX_MONTHS'] = 0 if NEW_ACCTS_IN_LAST_SIX_MONTHS == 0 else 1
    single_row_df['PRI_OVERDUE_ACCTS'] = 0 if PRI_OVERDUE_ACCTS == 0 else 1
    single_row_df['NO_OF_INQUIRIES'] = 0 if NO_OF_INQUIRIES == 0 else 1

    #Outlier Clipping
    single_row_df['ASSET_COST'] = ASSET_COST
    single_row_df['ASSET_COST'] = single_row_df['ASSET_COST'].clip(0, 0.3 * 1e6)

    single_row_df['DISBURSED_AMOUNT'] = DISBURSED_AMOUNT
    single_row_df['DISBURSED_AMOUNT'] = single_row_df['DISBURSED_AMOUNT'].clip(0, 0.2 * 1e6)

    single_row_df['PRI_CURRENT_BALANCE'] = PRI_CURRENT_BALANCE
    single_row_df['PRI_CURRENT_BALANCE'] = single_row_df['PRI_CURRENT_BALANCE'].clip(0, 0.4 * 1e8)

    single_row_df['PRI_DISBURSED_AMOUNT'] = PRI_DISBURSED_AMOUNT
    single_row_df['PRI_DISBURSED_AMOUNT'] = single_row_df['PRI_DISBURSED_AMOUNT'].clip(0, 4 * 1e7)

    single_row_df['PRIMARY_INSTAL_AMT'] = PRIMARY_INSTAL_AMT
    single_row_df['PRIMARY_INSTAL_AMT'] = single_row_df['PRIMARY_INSTAL_AMT'].clip(0, 3 * 1e6)

    single_row_df['AVERAGE_ACCT_AGE'] = AVERAGE_ACCT_AGE
    single_row_df['AVERAGE_ACCT_AGE'] = single_row_df['AVERAGE_ACCT_AGE'].clip(0, 200)

    single_row_df['CREDIT_HISTORY_LENGTH'] = CREDIT_HISTORY_LENGTH
    single_row_df['CREDIT_HISTORY_LENGTH'] = single_row_df['CREDIT_HISTORY_LENGTH'].clip(0, 300)

    single_row_df['LTV'] = LTV
    single_row_df['LTV'] = single_row_df['LTV'].clip(46.69, 105.85)


    single_row_df['SEC_CURRENT_BALANCE'] = SEC_CURRENT_BALANCE

    single_row_df['SEC_DISBURSED_AMOUNT'] = SEC_DISBURSED_AMOUNT

    single_row_df['PERFORM_CNS_SCORE'] = PERFORM_CNS_SCORE

    single_row_df['SEC_INSTAL_AMT'] = SEC_INSTAL_AMT

    single_row_df['AVERAGE_ACCT_AGE'] = AVERAGE_ACCT_AGE

    single_row_df['CREDIT_HISTORY_LENGTH'] = CREDIT_HISTORY_LENGTH

    single_row_df['DISBURSAL_AGE'] = DISBURSAL_DATE.year - DATE_OF_BIRTH.year - ((DISBURSAL_DATE.month, DISBURSAL_DATE.day) < (DATE_OF_BIRTH.month, DATE_OF_BIRTH.day))

    single_row_df['AADHAR_FLAG'] = AADHAR_FLAG

    single_row_df['VOTERID_FLAG'] = VOTERID_FLAG

    num_type_cols =  ['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV', 
                      'PERFORM_CNS_SCORE','PRI_CURRENT_BALANCE', 'PRI_DISBURSED_AMOUNT','SEC_CURRENT_BALANCE', 'SEC_DISBURSED_AMOUNT', 
                      'PRIMARY_INSTAL_AMT',  'SEC_INSTAL_AMT', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'DISBURSAL_AGE']


    selected_columns_to_scale = [
    "DISBURSED_AMOUNT", "ASSET_COST", "LTV", "PERFORM_CNS_SCORE", 
    "PRI_CURRENT_BALANCE", "PRI_DISBURSED_AMOUNT", "SEC_CURRENT_BALANCE", 
    "SEC_DISBURSED_AMOUNT", "PRIMARY_INSTAL_AMT", "SEC_INSTAL_AMT", 
    "AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH", "DISBURSAL_AGE"
    ]

    print(single_row_df.transpose())

    df_for_scaler = single_row_df[selected_columns_to_scale].copy()

    df_num_after_scale = scaler.transform(df_for_scaler)

    # Assigning the scaled values back to the original dataframe
    single_row_df[selected_columns_to_scale] = df_num_after_scale

    print( df_num_after_scale.transpose())

    print(single_row_df.transpose())

    print("model ", model_choice )

    if model_choice == "KNN":
        prediction = knn.predict(single_row_df)
        prediction_value = prediction[0]

    if model_choice == "XGBoost":
        prediction = xgb.predict(single_row_df)
        prediction_value = prediction[0]

    if model_choice == "Random Forest":
        prediction = rfc.predict(single_row_df)
        prediction_value = prediction[0]

    if model_choice == "Neural Network":
        prediction = nn_model.predict(single_row_df)
        threshold = 0.5
        prediction_after_threshold = [1 if p >= threshold else 0 for p in prediction]
        prediction_value = prediction_after_threshold[0]


    return prediction_value

def main():

    # UI Form for data input

    col1, col2, col3, col4 = form_column.columns(4)
    UNIQUEID = col1.text_input("UNIQUEID", value="644253")
    DISBURSED_AMOUNT = col2.number_input("DISBURSED_AMOUNT", value=48428)
    ASSET_COST = col3.number_input("ASSET_COST", value=67364)
    LTV = col4.number_input("LTV", value=74.74)
    BRANCH_ID = col1.text_input("BRANCH_ID", value="249")
    SUPPLIER_ID = col2.text_input("SUPPLIER_ID", value="21479")
    MANUFACTURER_ID = col3.text_input("MANUFACTURER_ID", value="45")
    CURRENT_PINCODE_ID = col4.text_input("CURRENT_PINCODE_ID", value="2226")
    DATE_OF_BIRTH = col1.date_input("DATE_OF_BIRTH", value=datetime.datetime.strptime("19-06-1997", "%d-%m-%Y").date())
    EMPLOYMENT_TYPE = col2.selectbox("EMPLOYMENT_TYPE", options=['Salaried', 'Self employed', 'Unclassified'])
    DISBURSAL_DATE = col3.date_input("DISBURSAL_DATE", value=datetime.datetime.strptime("30-10-2018", "%d-%m-%Y").date())
    STATE_ID = col4.text_input("STATE_ID", value="4")
    EMPLOYEE_CODE_ID = col1.text_input("EMPLOYEE_CODE_ID", value="2520")
    MOBILENO_AVL_FLAG = col2.radio("MOBILENO_AVL_FLAG", options=[0,1])
    AADHAR_FLAG = col3.radio("AADHAR_FLAG", options=[0,1])
    PAN_FLAG = col4.radio("PAN_FLAG", options=[0,1])
    VOTERID_FLAG = col1.radio("VOTERID_FLAG", options=[0,1])
    DRIVING_FLAG = col2.radio("DRIVING_FLAG", options=[0,1])
    PASSPORT_FLAG = col3.radio("PASSPORT_FLAG", options=[0,1])
    PERFORM_CNS_SCORE = col4.number_input("PERFORM_CNS_SCORE", value=534, format="%d")
    PERFORM_CNS_SCORE_DESCRIPTION = col1.selectbox("PERFORM_CNS_SCORE_DESCRIPTION", options=['No Bureau History Available', 'I-Medium Risk', 'L-Very High Risk', 'A-Very Low Risk', 'Not Scored: Not Enough Info available on the customer', 'D-Very Low Risk', 'M-Very High Risk', 'B-Very Low Risk', 'C-Very Low Risk', 'E-Low Risk', 'H-Medium Risk', 'F-Low Risk', 'K-High Risk', 'Not Scored: No Activity seen on the customer (Inactive)', 'Not Scored: Sufficient History Not Available', 'Not Scored: No Updates available in last 36 months', 'G-Low Risk', 'J-High Risk', 'Not Scored: Only a Guarantor', 'Not Scored: More than 50 active Accounts found'])
    PRI_NO_OF_ACCTS = col2.number_input("PRI_NO_OF_ACCTS", value=1, format="%d")
    PRI_ACTIVE_ACCTS = col3.number_input("PRI_ACTIVE_ACCTS", value=1, format="%d")
    PRI_OVERDUE_ACCTS = col4.number_input("PRI_OVERDUE_ACCTS", value=1 , format="%d")
    PRI_CURRENT_BALANCE = col1.number_input("PRI_CURRENT_BALANCE", value=0, format="%d")
    PRI_SANCTIONED_AMOUNT = col2.number_input("PRI_SANCTIONED_AMOUNT", value=0, format="%d")
    PRI_DISBURSED_AMOUNT = col3.number_input("PRI_DISBURSED_AMOUNT", value=0, format="%d")
    SEC_NO_OF_ACCTS = col4.number_input("SEC_NO_OF_ACCTS", value=0, format="%d")
    SEC_ACTIVE_ACCTS = col1.number_input("SEC_ACTIVE_ACCTS", value=0, format="%d")
    SEC_OVERDUE_ACCTS = col2.number_input("SEC_OVERDUE_ACCTS", value=0, format="%d")
    SEC_CURRENT_BALANCE = col3.number_input("SEC_CURRENT_BALANCE", value=0, format="%d")
    SEC_SANCTIONED_AMOUNT = col4.number_input("SEC_SANCTIONED_AMOUNT", value=0, format="%d")
    SEC_DISBURSED_AMOUNT = col1.number_input("SEC_DISBURSED_AMOUNT", value=0, format="%d")
    PRIMARY_INSTAL_AMT = col2.number_input("PRIMARY_INSTAL_AMT", value=0, format="%d")
    SEC_INSTAL_AMT = col3.number_input("SEC_INSTAL_AMT", value=0, format="%d")
    NEW_ACCTS_IN_LAST_SIX_MONTHS = col4.number_input("NEW_ACCTS_IN_LAST_SIX_MONTHS", value=0, format="%d")
    DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS = col1.number_input("DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", value=0, format="%d")
    AVERAGE_ACCT_AGE = col2.number_input("AVERAGE_ACCT_AGE (number of months)", value=50, format="%d" )
    CREDIT_HISTORY_LENGTH = col3.number_input("CREDIT_HISTORY_LENGTH (number of months)", value=50, format="%d")
    NO_OF_INQUIRIES = col4.number_input("NO_OF_INQUIRIES", value=0, format="%d")

    model_choice = image_column.radio(
        "Choose a model",
        ("KNN", "XGBoost", "Random Forest", "Neural Network")
    )

    if image_column.button('Submit'):
        # Call the model with the form values
        
        prediction = predict_loan_default(
        UNIQUEID, DISBURSED_AMOUNT, ASSET_COST, LTV, BRANCH_ID, SUPPLIER_ID, 
        MANUFACTURER_ID, CURRENT_PINCODE_ID, DATE_OF_BIRTH, EMPLOYMENT_TYPE, 
        DISBURSAL_DATE, STATE_ID, EMPLOYEE_CODE_ID, MOBILENO_AVL_FLAG, AADHAR_FLAG, 
        PAN_FLAG, VOTERID_FLAG, DRIVING_FLAG, PASSPORT_FLAG, PERFORM_CNS_SCORE, 
        PERFORM_CNS_SCORE_DESCRIPTION, PRI_NO_OF_ACCTS, PRI_ACTIVE_ACCTS, 
        PRI_OVERDUE_ACCTS, PRI_CURRENT_BALANCE, PRI_SANCTIONED_AMOUNT, 
        PRI_DISBURSED_AMOUNT, SEC_NO_OF_ACCTS, SEC_ACTIVE_ACCTS, SEC_OVERDUE_ACCTS, 
        SEC_CURRENT_BALANCE, SEC_SANCTIONED_AMOUNT, SEC_DISBURSED_AMOUNT, 
        PRIMARY_INSTAL_AMT, SEC_INSTAL_AMT, NEW_ACCTS_IN_LAST_SIX_MONTHS, 
        DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS, AVERAGE_ACCT_AGE, 
        CREDIT_HISTORY_LENGTH, NO_OF_INQUIRIES, model_choice
    )
        image_column.write(f"Predicted Loan Default Result: {prediction}")



if __name__=='__main__':
    main()
