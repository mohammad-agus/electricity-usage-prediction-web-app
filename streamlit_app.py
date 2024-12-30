import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('⚡Electricity Usage Prediction')
st.info('The linear regression model in this web app is primarily built and evaluated using numpy, pandas, and seaborn libraries.')


df = pd.read_csv('steel_industry_energy_consumption.csv')
del df['Unnamed: 0']

# Convert columns name to lowercase and replace period to underscore
df.columns = df.columns.str.lower().str.replace(".", "_")

# Get categorical columns name
categorical = list(df.dtypes[(df.dtypes == 'object')].index)

# Convert columns value to lowercase
for col in categorical:
    df[col] = df[col].str.lower()

# Get numerical columns name
numerical = list(df.dtypes[(df.dtypes != 'object')].index)

# Features columns
numerical_f = numerical.copy()
numerical_f.remove('usage_kwh')
features = categorical + numerical_f

with st.expander('Dataset'):
    st.write('**Description**')
    st.markdown('\
                This app utilizes the \
                [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) \
                dataset from UCI machine learning repository website.')
    st.markdown(
            """
                **Target (dependent variable): usage_kwh** \n
                **Features (independent variables):**
                * lagging_current_reactive_power_kvarh
                * leading_current_reactive_power_kvarh
                * co2(tco2)
                * lagging_current_power_factor
                * leading_current_power_factor
                * nsm: number of seconds from midnight
                * weekstatus
                * day_of_week
                * load_type
            """
        )
    st.dataframe(df)

sns.set(font_scale=0.8)

with st.expander('Correlation heatmap of the numerical features and the target'):
    fig1, ax1 = plt.subplots()
    sns.heatmap(df[numerical].corr(), ax=ax1, annot=True)
    st.write(fig1)


with st.expander('Box plot of a categorical feature and the target'):
    option = st.selectbox(
        "Categorical Feature",
        categorical,
        index=None,
        placeholder="select a feature.."
    )
    if option:
        fig2, ax2 = plt.subplots()
        sns.boxenplot(y=df['usage_kwh'], x=df[option], ax=ax2)
        st.write(fig2)
    pass



