import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Electricity Usage Prediction",
    page_icon="🔎",
)


#st.sidebar.success("Select a demo above.")


st.title('⚡Electricity Usage Prediction')


st.markdown('##### Objective')
st.write('To simulate a machine learning model as a web app that predicts electricity consumption (kilo-Watt-hour). \
         The linear regression model in this web app is primarily implemented and evaluated using numpy, pandas, and seaborn libraries.')
st.write('This app utilizes the Steel Industry Energy Consumption dataset from UCI machine learning repository website.')
st.link_button("Go to data source", "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption")







