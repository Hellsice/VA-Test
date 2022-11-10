#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import plotly.express as px
import opendatasets as od
import requests
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[2]:
st.set_page_config(layout="wide", page_title='Disaster influence on economy', initial_sidebar_state='expanded')
st.title('Disaster influence on economy')
st.sidebar.title('Navigation')
pages = st.sidebar.radio('Pages', options=['Home', 'Data cleaning', 'Map', 'Economic change', 'Comparison disasters'])

countries_geojson = gpd.read_file('countries.geojson')
GDP = pd.read_excel('GDP.xls', skiprows=[0,1,2], engine='xlrd')
Population = pd.read_excel('Population.xls', skiprows=[0,1,2], engine='xlrd')


# In[3]:


df = countries_geojson.merge(GDP, left_on='ISO_A3', right_on='Country Code', how='left')



# In[5]:


rampen_df = pd.read_csv('rampen_df.csv')



# In[6]:

with st.form(key='my_form'):
    Total_affected_mult = st.slider('Set the total affected multiplier',min_value=0.0, value=0.3 ,max_value=1.0, step=0.01)
    Intensity_threshold = st.number_input('Set the intensity threshold (default: 0.00001)', min_value=0.0, value=0.00001, max_value=1.0, step=0.00001)
    jaar = st.slider('Select year',min_value=1961, value=2018 ,max_value=2018)
    commit = st.form_submit_button('Submit')

    
