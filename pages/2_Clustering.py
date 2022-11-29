import streamlit as st
import importlib
module = importlib.import_module('pages.1_Regresi')
input_user = module.input_user

st.set_page_config(layout='wide')
st.title('Clustering')

df = input_user()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')