import pandas as pd
import geopandas as gpd
import plotly.express as px
import opendatasets as od
import requests
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(layout="wide", page_title='Disaster influence on economy', initial_sidebar_state='expanded')
st.title('Disaster influence on economy')
st.sidebar.title('Navigation')
pages = st.sidebar.radio('Pages', options=['Home', 'Data cleaning', 'Map', 'Economic change', 'Comparison disasters', 'The Big 4'])

od.download('https://datahub.io/core/geo-countries/r/countries.geojson')
countries_geojson = gpd.read_file('countries.geojson')
response = requests.get('https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=excel')
output = open('GDP.xls', 'wb')
output.write(response.content)
output.close()
GDP = pd.read_excel('GDP.xls', skiprows=[0,1,2], engine='xlrd')

response = requests.get('https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel')
output = open('Population.xls', 'wb')
output.write(response.content)
output.close()
Population = pd.read_excel('Population.xls', skiprows=[0,1,2], engine='xlrd')



df = countries_geojson.merge(GDP, left_on='ISO_A3', right_on='Country Code', how='left')


rampen_df = pd.read_csv('rampen_df.csv')




if pages== 'Map' or pages == 'Economic change' or pages == 'Comparison disasters' or pages == 'The Big 4':
    with st.form(key='my_form'):
        commit = st.form_submit_button('Submit')
        Total_affected_mult = st.slider('Set the total affected multiplier',min_value=0.0, value=0.3 ,max_value=1.0, step=0.01)
        Intensity_threshold = st.number_input('Set the intensity threshold (default: 0.00001)', min_value=0.0, value=0.00001, max_value=1.0, step=0.00001)
        jaar = st.slider('Select year',min_value=1961, value=2018 ,max_value=2018)
        
        
        round_mult = 100000
        rampen_df['Intensity'] = 0
        for i in range(len(rampen_df)):
            a = Population[Population['Country Code']==rampen_df['ISO'][i]]
            if len(a) == 1:
                rampen_df['Intensity'][i] = (rampen_df['Total Deaths'][i]+Total_affected_mult*rampen_df['Total Affected new'][i])/(a[str(rampen_df['Year'][i])].values[0])


        rampen_df = rampen_df[rampen_df['Intensity'] >= Intensity_threshold].reset_index(drop=True)
        rampen_df = rampen_df[['Year', 'ISO', 'Country', 'Disaster Group', 'Disaster Subgroup', 'Disaster Type',
               'Disaster Subtype', 'Total Deaths', 'Total Affected new', 'Intensity', 'Jaar 0', 'Jaar 1', 'Jaar 2', 'Jaar 3']]




        quantiles_subtypes = rampen_df.groupby(['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'])['Intensity'].quantile([0.25, 0.75]).reset_index()
        quantiles_subtypes = quantiles_subtypes.pivot(index=['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'], 
                                    columns = 'level_4', values='Intensity').reset_index()
        test = rampen_df.merge(quantiles_subtypes, left_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'], 
                               right_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'], how='left')
        test.columns = ['Year','ISO','Country','Disaster Group','Disaster Subgroup','Disaster Type',
                        'Disaster Subtype','Total Deaths','Total Affected new','Intensity','Jaar 0','Jaar 1','Jaar 2','Jaar 3',"0.25",'0.75']
        test['Category Subtypes']=0
        for index, row in test.iterrows():
            test.iloc[index, test.columns.get_loc('Category Subtypes')] = 3 if row['Intensity']>row['0.75'] else (1 if row['Intensity']<row['0.25'] else 2)

        
        
        quantiles_types = rampen_df.groupby(['Disaster Group', 'Disaster Subgroup', 'Disaster Type'])['Intensity'].quantile([0.25, 0.75]).reset_index()
        quantiles_types = quantiles_types.pivot(index=['Disaster Group', 'Disaster Subgroup', 'Disaster Type'], 
                                    columns = 'level_3', values='Intensity').reset_index()
        test2 = rampen_df.merge(quantiles_types, left_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type'], 
                               right_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type'], how='left')
        test2.columns = ['Year','ISO','Country','Disaster Group','Disaster Subgroup','Disaster Type', 'Disaster Subtype',
                 'Total Deaths','Total Affected new','Intensity','Jaar 0','Jaar 1','Jaar 2','Jaar 3',"0.25",'0.75']
        test2['Category Types']=0
        for index, row in test2.iterrows():
            test2.iloc[index, test2.columns.get_loc('Category Types')] = 3 if row['Intensity']>row['0.75'] else (1 if row['Intensity']<row['0.25'] else 2)
        
        
        rampen_df['Category Types'] = test2['Category Types']
        rampen_df['Category Subtypes'] = test['Category Subtypes']
        rampen_df = rampen_df.sort_values(by='Year').reset_index(drop=True)
        
        
        
        if pages == 'Map':
            Soort_data = ''
            Soort_data = st.selectbox('Select data type', ['Intensity', 'Affected'])
            Soort_data_dict = {'Intensity':'Intensity','Affected':'Affected'}
        if pages == 'Economic change':
            landen = rampen_df[['Country', 'ISO']]
            landen_naam = np.sort(rampen_df['Country'].unique())
            landen_dict = landen.set_index('Country').to_dict()
            landen_dict = landen_dict['ISO']
            landen_box = st.selectbox('Choose a country', landen_naam)
            land_code= landen_dict[landen_box]
        if pages == 'Comparison disasters':
            types = np.sort(list(rampen_df['Disaster Subtype'].unique()))  
            type_names = list(rampen_df['Disaster Subtype'].unique())
            type_dict = dict(zip(types, type_names))
            type_box=st.selectbox('Choose a subtype', types)
            
            Outlier_box = st.selectbox('Remove outliers', ['No', 'Yes'])
        if pages == 'The Big 4':
            categories = ['Categorie 1', 'Categorie 2', 'Categorie 3']
            category= ['Category 1', 'Category 2', 'Category 3']
            category_dict = dict(zip(categories, category))
            category_box = st.selectbox('Choose a disaster category', categories)
  
               

if pages == 'Map':
    if Soort_data_dict[Soort_data]=='Intensity':
        rampen_df_intensity = rampen_df.groupby(['ISO', 'Country', 'Year'])['Intensity'].max().to_frame().reset_index()
        rampen_df_intensity = rampen_df_intensity.pivot_table(
            index=['ISO', 'Country'], 
            columns='Year', 
            values='Intensity'
        ).reset_index()
        rampen_df_intensity.index.name = rampen_df_intensity.columns.name = None
        rampen_df_intensity = rampen_df_intensity.fillna(0)
        rampen_df_intensity = countries_geojson.merge(rampen_df_intensity, left_on='ISO_A3', right_on='ISO', how='right')
        df_adjusted = rampen_df_intensity
    elif Soort_data_dict[Soort_data]=='Affected':
        rampen_df_affected = rampen_df.groupby(['ISO', 'Country', 'Year'])['Total Affected new'].sum().to_frame().reset_index()
        rampen_df_affected = rampen_df_affected.pivot_table(
            index=['ISO', 'Country'], 
            columns='Year', 
            values='Total Affected new'
        ).reset_index()
        rampen_df_affected.index.name = rampen_df_affected.columns.name = None
        rampen_df_affected = rampen_df_affected.fillna(0)
        rampen_df_affected = countries_geojson.merge(rampen_df_affected, left_on='ISO_A3', right_on='ISO', how='right')
        df_adjusted = rampen_df_affected

    map = px.choropleth_mapbox(df_adjusted, geojson=df_adjusted.geometry, locations=df_adjusted.index, color=jaar, mapbox_style="open-street-map",
                          hover_name=df_adjusted.Country, zoom=1, height=800)
    st.plotly_chart(map, use_container_width=True)


if pages == 'Economic change':
    col5, col6 = st.columns([1,1])
    land = landen_box
    check = rampen_df[rampen_df['Country']==landen_box]
    check = check[check['Year']==jaar]
    
    if len(check) == 0:
        a = 'Er was geen ramp in dit jaar'
        #st.markdown(a)
    else:
        type_rampen = []
        subtypen = []
        intensiteit=[]
        for i in range(len(check)):
            type_rampen.append(check['Disaster Type'].values[i])
            subtypen.append(check['Disaster Subtype'].values[i])
            intensiteit.append(check['Intensity'].values[i])
            if subtypen == 0:
                a = 'In ' + str(jaar) + ' was er in ' + landen_box + ' een natuurramp met type: ' + type_rampen[i]\
                +'.\nHiervan was de intensiteit: ' + str(round(intensiteit[i]*round_mult)/round_mult)
            else:
                a = 'In ' + str(jaar) + ' was er in ' + landen_box + ' een natuurramp met type: ' + type_rampen[i] + ' en subtype ' + subtypen[i]\
                +'.\nHiervan was de intensiteit: ' + str(round(intensiteit[i]*round_mult)/round_mult)
            print(a)
            with col5:
                st.markdown(a)

    
    grafiek_max_jaar = jaar+5
    grafiek_min_jaar = jaar-2
    GDP_grafiek = GDP.drop(['Country Name', 'Indicator Name', 'Indicator Code'], axis=1)
    GDP_grafiek = GDP_grafiek.set_index('Country Code').T.rename(pd.to_numeric).reset_index()
    GDP_grafiek = GDP_grafiek[(GDP_grafiek['index']>=grafiek_min_jaar) & (GDP_grafiek['index']<=grafiek_max_jaar)]
    GDP_grafiek = GDP_grafiek.rename(columns={'index':'Year'}, index={'Country Code':'Index'})
    GDP_land = GDP_grafiek[['Year', land_code, 'WLD']]
    with col5:
        GDP_land

    GDP_fig = make_subplots(specs=[[{"secondary_y": True}]])
    GDP_fig.add_trace(
        go.Line(x=GDP_grafiek['Year'].to_list(), y=GDP_grafiek[land_code].to_list(), name=landen_box),
        secondary_y=False)
    GDP_fig.add_trace(
        go.Line(x=GDP_grafiek['Year'].to_list(), y=GDP_grafiek['WLD'].to_list(), name="World"),
        secondary_y=True)
    GDP_fig.update_layout(
        title_text="<b>GDP comparising of world vs. " + landen_box +'</b>')
    GDP_fig.update_xaxes(title_text="<b>2 years before and 5 years after chosen year</b>")
    GDP_fig.update_yaxes(title_text='<b>GDP ' + landen_box + '</b>', secondary_y=False)
    GDP_fig.update_yaxes(title_text='<b>GDP world</b>', secondary_y=True)
    with col6:
        st.plotly_chart(GDP_fig)




if pages == 'Comparison disasters': 
    col3, col4 = st.columns([1,1])
    data_subtypes = rampen_df[rampen_df['Disaster Subtype']==type_box].reset_index(drop=True)
    Fig_subtypes_jaar0 = px.box(data_subtypes, x='Category Subtypes', y='Jaar 0', color='Category Subtypes')
    Fig_subtypes_jaar0.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 0</b>')
    Fig_subtypes_jaar0.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar0.update_xaxes(title = '<b>Category</b>')
    
    with col3:
        st.plotly_chart(Fig_subtypes_jaar0)
    
    
    Fig_subtypes_jaar1 = px.box(data_subtypes, x='Category Subtypes', y='Jaar 1', color='Category Subtypes')
    Fig_subtypes_jaar1.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 1</b>')
    Fig_subtypes_jaar1.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar1.update_xaxes(title = '<b>Category</b>')
    
    with col4:
        st.plotly_chart(Fig_subtypes_jaar1)
    
    
    Fig_subtypes_jaar2 = px.box(data_subtypes, x='Category Subtypes', y='Jaar 2', color='Category Subtypes')
    Fig_subtypes_jaar2.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 2</b>')
    Fig_subtypes_jaar2.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar2.update_xaxes(title = '<b>Category</b>')
    
    with col3:
        st.plotly_chart(Fig_subtypes_jaar2)
    
    
    Fig_subtypes_jaar3 = px.box(data_subtypes, x='Category Subtypes', y='Jaar 3', color='Category Subtypes')
    Fig_subtypes_jaar3.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 3</b>')
    Fig_subtypes_jaar3.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar3.update_xaxes(title = '<b>Category</b>')
    
    with col4:
        st.plotly_chart(Fig_subtypes_jaar3)
    
    

if pages == 'The Big 4':
    if category_dict[category_box] == 'Category 1':
        Category_data = rampen_df[(rampen_df['Category Types']==1)]
        Category_data = Category_data[(Category_data['Disaster Type']=='Drought') | (Category_data['Disaster Type']=='Flood') | (Category_data['Disaster Type']=='Storm') | (Category_data['Disaster Type']=='Earthquake')]
        Category_data = Category_data.dropna(axis=0)
        Category_data = Category_data.sort_values(by='Disaster Type').reset_index(drop=True)
    elif category_dict[category_box] == 'Category 2':
        Category_data = rampen_df[(rampen_df['Category Types']==2)]
        Category_data = Category_data[(Category_data['Disaster Type']=='Drought') | (Category_data['Disaster Type']=='Flood') | (Category_data['Disaster Type']=='Storm') | (Category_data['Disaster Type']=='Earthquake')]
        Category_data = Category_data.dropna(axis=0)
        Category_data = Category_data.sort_values(by='Disaster Type').reset_index(drop=True)
    elif category_dict[category_box] == 'Category 3':
        Category_data = rampen_df[(rampen_df['Category Types']==3)]
        Category_data = Category_data[(Category_data['Disaster Type']=='Drought') | (Category_data['Disaster Type']=='Flood') | (Category_data['Disaster Type']=='Storm') | (Category_data['Disaster Type']=='Earthquake')]
        Category_data = Category_data.dropna(axis=0)
        Category_data = Category_data.sort_values(by='Disaster Type').reset_index(drop=True)
    
    
    fig_a2 = px.box(Category_data, x='Disaster Type', y='Jaar 0', color='Disaster Type')
    fig_b2 = px.box(Category_data, x='Disaster Type', y='Jaar 1', color='Disaster Type')
    fig_c2 = px.box(Category_data, x='Disaster Type', y='Jaar 2', color='Disaster Type')
    fig_d2 = px.box(Category_data, x='Disaster Type', y='Jaar 3', color='Disaster Type')
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig_a2)
        st.plotly_chart(fig_c2)
    with col2:
        st.plotly_chart(fig_b2)
        st.plotly_chart(fig_d2)
