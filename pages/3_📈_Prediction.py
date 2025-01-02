import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


st.set_page_config(page_title="Prediction", page_icon="📈")
st.title("📈Prediction")


df = pd.read_csv('steel_industry_energy_consumption.csv')
del df['Unnamed: 0']
df.columns = df.columns.str.lower().str.replace(".", "_")
del df['weekstatus']
categorical = list(df.dtypes[(df.dtypes == 'object')].index)
for col in list(df.dtypes[(df.dtypes == 'object')].index):
    df[col] = df[col].str.lower()
del df['usage_kwh']






with st.sidebar:
    st.markdown("### Input variabels:")
    lagging_current_reactive_power_kvarh = st.slider(
        label="Lagging current reactive power (kVarh)",
        min_value=0.00,
        max_value=97.00,
        value=0.00,
        step=0.01
    )

    leading_current_reactive_power_kvarh = st.slider(
        label="Leading current reactive power (kVarh)",
        min_value=0.00,
        max_value=30.00,
        value=0.00,
        step=0.01
    )

    co2_tco2 = st.slider(
        label="CO2(tCO2)",
        min_value=0.00,
        max_value=0.1,
        step=0.01
    )

    lagging_current_power_factor = st.slider(
        label="Lagging current power factor",
        min_value=0.00,
        max_value=100.00,
        value=0.00,
        step=0.01
    )

    leading_current_power_factor = st.slider(
        label="Leading current power factor",
        min_value=0.00,
        max_value=100.00,
        value=0.00,
        step=0.01
    )

    nsm = st.slider(
        label="Number of seconds from midnight",
        min_value=0,
        max_value=90000,
        value=0,
        step=60
    )

    dayofweek = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_of_week = st.pills(
        label="Day of week",
        options=dayofweek,
        selection_mode='single'
    )
    
    loadtypes = ['light_load', 'medium_load', 'maximum_load']
    load_type = st.pills(
        label="Load type",
        options=loadtypes,
        selection_mode='single'
    )


with open('lr_model_regularized_full.pickle', 'rb') as f:
    intercept, coef = pickle.load(f)

cols = df.columns.to_list()
input_variabels = [
   lagging_current_reactive_power_kvarh, leading_current_reactive_power_kvarh,
   co2_tco2, lagging_current_power_factor, leading_current_power_factor, nsm,
day_of_week, load_type
]

st.write(" ")

if None in input_variabels:
    st.error("Input all variabels!", icon="❗")
else:
    st.write("***Input variabels:***")
    input_vars_df = pd.DataFrame(data=[input_variabels], columns=cols)
    df_w_input = pd.concat([input_vars_df, df])

    x = pd.get_dummies(df_w_input, dtype=int)
    predictions = intercept + x.iloc[0:1].values @ coef

    st.table(df_w_input.iloc[0:1])
    st.metric(label='Electricity usage', value=f'{round(predictions[0],3)} kWh')