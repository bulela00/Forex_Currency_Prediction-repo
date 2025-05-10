from autots import AutoTS
import lightgbm as lgb
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')


import streamlit as st
st.title(("Future Forex Currency Price Prediction Model"))

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    #'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZEALAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN$': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON$': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN$': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

def get_recent_target_series(data, selected_option, n_lags=10):
    target_col = options[selected_option]
    series = data[['Time Serie', target_col]].dropna()
    series.columns = ['Time Serie', 'target']
    return series[['target']].tail(n_lags).reset_index(drop=True)

#function to make predictions, we'll use the code from analysis.ipynb file and make a function which would return forecasts
def make_forecast(selected_option,forecast):
    # Load data 
    data = pd.read_csv("data/Foreign_Exchange_Rates.xls")
    print(data.head())

    # Data preprocessing
    data.drop(['Unnamed: 0', 'Unnamed: 24'], axis=1, inplace=True)
    data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')   # Convert timeseeries to datetime
    # Convert columns to numeric data
    obj_cols = data.columns.to_list()
    obj_cols.remove('Time Serie')
    for col in obj_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')   # If a value is not a number NaN will be returned



    # Loading pretrained model based on selected option
        #'AUSTRALIAN DOLLAR': LGB 
    if selected_option == 'AUSTRALIAN DOLLAR':
        model = joblib.load( "models/AUSD_model.pkl")
        mod_type = 1 # LGB

        #'EURO': AutoTS 
    elif selected_option == 'EURO':   
        with open("models/EU_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'GREAT BRITAIN POUNDS': LGB 
    elif selected_option == 'GREAT BRITAIN POUNDS':     
        model = joblib.load("models/GBP_model.pkl")
        mod_type = 1 # LGB


        #'BRAZILIAN REAL': LGB
    elif selected_option == 'BRAZILIAN REAL':
        model = joblib.load("models/BR_model.pkl")
        mod_type = 1 # LGB

        #'CANADIAN DOLLAR': AutoTS 
    elif selected_option == 'CANADIAN DOLLAR':
        with open("models/CD_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'CHINESE YUAN$': LGB 
    elif selected_option == 'CHINESE YUAN$':
        model = joblib.load("models/CHI_model.pkl")
        mod_type = 1 # LGB

        #'HONG KONG DOLLAR': LGB 
    elif selected_option == 'HONG KONG DOLLAR':
        model = joblib.load("models/HKD_model.pkl")
        mod_type = 1 # LGB

        #'INDIAN RUPEE': AutoTS 
    elif selected_option == 'INDIAN RUPEE':
        with open("models/IR_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'KOREAN WON$': AutoTS 
    elif selected_option == 'KOREAN WON$':
        with open("models/KW_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'MEXICAN PESO': LGB 
    elif selected_option == 'MEXICAN PESO':
        model = joblib.load("models/MP_model.pkl")
        mod_type = 1 # LGB

        #'SOUTH AFRICAN RAND$': LGB 
    elif selected_option == 'SOUTH AFRICAN RAND$':
        model = joblib.load("models/ZAR_model.pkl")
        mod_type = 1 # LGB

        #'SINGAPORE DOLLAR': AutoTS
    elif selected_option == 'SINGAPORE DOLLAR':
        with open("models/SD_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'DANISH KRONE': AutoTS 
    elif selected_option == 'DANISH KRONE':
        with open("models/DK_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'JAPANESE YEN$': AutoTS 
    elif selected_option == 'JAPANESE YEN$':
        with open("models/JY_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'MALAYSIAN RINGGIT': AutoTS 
    elif selected_option == 'MALAYSIAN RINGGIT':
        with open("models/MR_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS

        #'NORWEGIAN KRONE': LGB 
    elif selected_option == 'NORWEGIAN KRONE':
        model = joblib.load("models/NK_model.pkl")
        mod_type = 1 # LGB

        #'SWEDEN KRONA': LGB
    elif selected_option == 'SWEDEN KRONA': 
        model = joblib.load("models/SK_model.pkl")
        mod_type = 1 # LGB

        #'SRILANKAN RUPEE': LGB 
    elif selected_option == 'SRILANKAN RUPEE':
        model = joblib.load("models/SR_model.pkl")
        mod_type = 1 # LGB

        #'SWISS FRANC': LGB 
    elif selected_option == 'SWISS FRANC':
        model  = joblib.load("models/SF_model.pkl")
        mod_type = 1 # LGB

        #'NEW TAIWAN DOLLAR': AutoTS
    elif selected_option == 'NEW TAIWAN DOLLAR': 
        with open("models/NTD_model.pkl", "rb") as f:
            model = pickle.load(f)
            mod_type = 0 # AutoTS


        #'THAI BAHT': LGB
    elif selected_option == 'THAI BAHT': 
        model = joblib.load("models/TB_model.pkl")
        mod_type = 1 # LGB

    # Forecasting using selected model for inputed forecast length.
    if mod_type == 0:
        prediction = model.predict()
        forecast = prediction.forecast
    elif mod_type == 1:
        recent_history = get_recent_target_series(data, selected_option)
        forecast = forecast_with_lightgbm(model, recent_history,n_forecast=forecast)

    return forecast

    #currently the model is trained on every submit action from streamlit, find a solution to this problem so that on every submit action, a pretrained model for each currecncy is loaded and inferenced.


with st.form(key='user_form'):
    # Add input widgets to the form
    # Create the selectbox
    selected_option = st.selectbox('Choose a currency:', options)
    forecast = st.number_input(
    "Enter an integer",  # Label displayed to the user
    min_value=1,         # Minimum value allowed
    max_value=100,      # Maximum value allowed
    value=1,            # Default value
    step=1              # Increment step
)
    submit_button = st.form_submit_button(label='Generate Predictions')   # Button to submit selections and generate

if submit_button:
    
    forecast = make_forecast(selected_option,forecast)
        
    st.write(forecast)
    st.line_chart(forecast)
    st.dataframe(forecast)