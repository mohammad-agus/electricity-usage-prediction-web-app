import streamlit as st


st.set_page_config(
    page_title="Electricity Usage Prediction",
    page_icon="⚡",
)


#st.sidebar.success("Select a demo above.")


st.title('⚡Electricity Usage Prediction')


st.markdown('##### Objective')
st.write('To simulate a machine learning model as a web app that predicts daily electricity consumption in kilo-Watt-hour (kWh). \
         The linear regression model in this web app is primarily implemented and evaluated using numpy, pandas, and seaborn libraries.')
st.write('This app utilizes the Steel Industry Energy Consumption dataset from UCI machine learning repository website.')
st.link_button("Go to data source", "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption")

st.link_button("Go to Github project repo", "https://github.com/mohammad-agus/electricity-usage-prediction-web-app")

st.write("")
st.markdown('##### Contact Information')
st.markdown(
    """ 
    - **Email** : mohammad_agus@outlook.com
    - **LinkedIn** : [in/moh-agus](https://www.linkedin.com/in/moh-agus/)
    - **Github** : [mohammad-agus](https://github.com/mohammad-agus)
    """
    )






