import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Explore Data", page_icon="🔎")

st.title("🔎Explore Data")


df = pd.read_csv('steel_industry_energy_consumption.csv')
del df['Unnamed: 0']

# Convert columns name to lowercase and replace period to underscore
df.columns = df.columns.str.lower().str.replace(".", "_")
del df['weekstatus']

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

 
sns.set(font_scale=0.8)

st.write("")
st.markdown("##### Dataset")
st.dataframe(df)

st.markdown("###### Summary")
st.dataframe(df.describe(include='all'))


st.markdown(
        """
            ###### Target (dependent variable): 
            * usage_kwh
            ###### Features (independent variables):
            * lagging_current_reactive_power_kvarh
            * leading_current_reactive_power_kvarh
            * co2(tco2)
            * lagging_current_power_factor
            * leading_current_power_factor
            * nsm: number of seconds from midnight
            * weekstatus
            * load_type
        """
    )

st.write("")
st.markdown('##### Correlation heatmap of the numerical features and the target')

selected_num = st.multiselect(
    "Select two or more variabels to show the correlation..",
    numerical,
    default=numerical
)

if selected_num:
    fig1, ax1 = plt.subplots()
    sns.heatmap(df[selected_num].corr(), ax=ax1, annot=True, cmap='viridis')
    st.write(fig1)
pass

st.write("")
st.markdown('##### Boxen plot of a categorical feature and the target')
categorical_feature = st.selectbox(
    "Categorical Feature",
    categorical,
    index=None,
    placeholder="select a feature.."
)
if categorical_feature:
    fig2, ax2 = plt.subplots()
    plt.title(f"usage_kwh by {categorical_feature}")
    sns.boxenplot(x=df[categorical_feature], y=df['usage_kwh'], ax=ax2, palette='viridis')
    plt.tight_layout()
    st.pyplot(fig2)
pass