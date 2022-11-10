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
pages = st.sidebar.radio('Pages', options=['Home', 'General code', 'Map', 'Economic change', 'Comparison disasters', 'The Big 4', 'Sources'])

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

if pages == 'Home':
    st.markdown('Gemaakt door Bart Sil Combee')
    st.image('disasters.png')
    
    
if pages == 'Sources':
    st.markdown('Data sources:')
    st.markdown('Population: worldbank.org, https://data.worldbank.org/indicator/NY.GDP.MKTP.CD')
    st.markdown('GDP: worldbank.org, https://data.worldbank.org/indicator/SP.POP.TOTL')
    st.markdown('Disaster Data: EMDAT-public, https://public.emdat.be/data')
    st.markdown('Geodata: datahub, https://datahub.io/core/geo-countries')
    st.markdown('Formula for Intensity: "The Growth Aftermath of Natural Disasters" by Thomas Fomby, Yuki Ikeda and Norman Loayza')


if pages == 'General code':
    st.title('General Code')
    st.markdown("Data retrieved via API's")
    st.code("response = requests.get('https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=excel') \n\
output = open('GDP.xls', 'wb')\n\
output.write(response.content)\n\
output.close()", language='python')
    st.markdown('')
    st.markdown('Removed unnecessary years.')
    st.code('rampen_df = rampen_df[rampen_df.Year >= 1961].reset_index(drop=True).fillna(0)\n\
rampen_df = rampen_df[rampen_df.Year <= 2021].reset_index(drop=True).fillna(0)', language='python')
    st.markdown('')
    st.markdown('Recalculated deaths and total affected.')
    st.code("rampen_df['Total Affected new'] = rampen_df['No Affected']+rampen_df['No Injured']+rampen_df['No Homeless']\n\
rampen_df_controle = rampen_df.groupby(['ISO', 'Country', 'Year', 'Disaster Group', 'Disaster Subgroup', 'Disaster Type', 'Disaster Subtype'])['Total Affected new'].sum().reset_index()\n\
rampen_df_controle2 = rampen_df.groupby(['ISO', 'Country', 'Year', 'Disaster Group', 'Disaster Subgroup', 'Disaster Type', 'Disaster Subtype'])['Total Deaths'].sum().reset_index()", language='python')
    st.markdown('')
    st.markdown('Determined GDP percentage change compared to the world.')
    st.code('''rampen_df.iloc[index, rampen_df.columns.get_loc('Jaar 1')] = \\
(GDP[GDP['Country Code']==rampen_df['ISO'][index]][str(rampen_df['Year'][index]+1)].values[0]\\
 - GDP[GDP['Country Code']==rampen_df['ISO'][index]][str(rampen_df['Year'][index])].values[0])/\\
GDP[GDP['Country Code']==rampen_df['ISO'][index]][str(rampen_df['Year'][index])].values[0]\\
 - (GDP[GDP['Country Code']=='WLD'][str(rampen_df['Year'][index]+1)].values[0]\\
 - GDP[GDP['Country Code']=='WLD'][str(rampen_df['Year'][index])].values[0])/\\
GDP[GDP['Country Code']=='WLD'][str(rampen_df['Year'][index])].values[0]''', language='python')
    st.code("rampen_df['Jaar 0']=rampen_df['Jaar 0']*100", language='python')
    st.markdown('')
    st.markdown("Making a submit button that's shared across different pages.")
    st.markdown('note, all code after this point is part of the button.')
    st.code('''with st.form(key='my_form'):
    commit = st.form_submit_button('Submit')
    Total_affected_mult = st.slider('Set the total affected multiplier',min_value=0.0, value=0.3 ,max_value=1.0, step=0.01)
    Intensity_threshold = st.number_input('Set the intensity threshold (default: 0.00001)', min_value=0.0, value=0.00001, max_value=1.0, step=0.00001)
    if pages == 'Map' or pages == 'Economic change':
        jaar = st.slider('Select year',min_value=1961, value=2018 ,max_value=2018)''',language='python')
    st.markdown('')
    st.markdown('Calculating intensity using the following formula:')
    st.markdown(r'''
$$ 
Intensity = \frac{Deaths + 0.3*Total\_Affected}{Population} 
$$
''')
    st.code('''rampen_df['Intensity'] = 0
for i in range(len(rampen_df)):
    a = Population[Population['Country Code']==rampen_df['ISO'][i]]
    if len(a) == 1:
        rampen_df['Intensity'][i] = (rampen_df['Total Deaths'][i]+Total_affected_mult*rampen_df['Total Affected new'][i])/(a[str(rampen_df['Year'][i])].values[0])''', language='python')
    st.markdown('Removing rows where intensity is below threshold.')
    st.code('''rampen_df = rampen_df[rampen_df['Intensity'] >= Intensity_threshold].reset_index(drop=True)''', language='python')
    st.markdown('')
    st.markdown('Calculating the quantiles of Disaster types and subtypes.')
    st.code('''quantiles_subtypes = rampen_df.groupby(['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'])['Intensity'].quantile([0.25, 0.75]).reset_index()
''', language='python')
    st.markdown('Pivoting quantiles dataframe.')
    st.code('''quantiles_subtypes = quantiles_subtypes.pivot(index=['Disaster Group', 'Disaster Subgroup', 'Disaster Type','Disaster Subtype'], columns = 'level_4', values='Intensity').reset_index()''', language='python')
    st.markdown('Merging dataframes and fixing column names.')
    st.code('''test2 = rampen_df.merge(quantiles_types, left_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type'], 
                       right_on=['Disaster Group', 'Disaster Subgroup', 'Disaster Type'], how='left')
test2.columns = ['Year','ISO','Country','Disaster Group','Disaster Subgroup','Disaster Type', 'Disaster Subtype','Total Deaths','Total Affected new','Intensity','Jaar 0','Jaar 1','Jaar 2','Jaar 3','0.25','0.75']''', language='python')
    st.markdown('Assigning category values to each disaster type and subtype.')
    st.code('''for index, row in rampen_df.iterrows():
    test2.iloc[index, test2.columns.get_loc('Category Types')] = 3 if row['Intensity']>row['0.75'] else (1 if row['Intensity']<row['0.25'] else 2)
''', language='python')
    st.code('''rampen_df['Category Types'] = test2['Category Types']
rampen_df = rampen_df.sort_values(by='Year').reset_index(drop=True)''', language='python')
    

    

   
    
    
    
    
    
    
    
if pages== 'Map' or pages == 'Economic change' or pages == 'Comparison disasters' or pages == 'The Big 4':
    with st.form(key='my_form'):
        commit = st.form_submit_button('Submit')
        Total_affected_mult = st.slider('Set the total affected multiplier',min_value=0.0, value=0.3 ,max_value=1.0, step=0.01)
        Intensity_threshold = st.number_input('Set the intensity threshold (default: 0.00001)', min_value=0.0, value=0.00001, max_value=1.0, step=0.00001)
        if pages == 'Map' or pages == 'Economic change':
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
        rampen_df_intensity.index.name = None
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
        rampen_df_affected.index.name = None
        rampen_df_affected = rampen_df_affected.fillna(0)
        rampen_df_affected = countries_geojson.merge(rampen_df_affected, left_on='ISO_A3', right_on='ISO', how='right')
        df_adjusted = rampen_df_affected

    map = px.choropleth_mapbox(df_adjusted, geojson=df_adjusted.geometry, locations=df_adjusted.index, color=jaar, mapbox_style="open-street-map",
                          hover_name=df_adjusted.Country, zoom=1, height=800)
    st.plotly_chart(map, use_container_width=True)
    
    st.title('Map specific code')
    st.markdown('Added a page specific selectbox to the submit button code with an if statement')
    st.code('''if pages == 'Map':
    Soort_data = st.selectbox('Select data type', ['Intensity', 'Affected'])''', language='python')
    st.markdown('')
    st.markdown('Filtered the dataset based on the selectbox choice. Intensity used max value, Affected used sum.')
    st.code('''rampen_df_intensity = rampen_df.groupby(['ISO', 'Country', 'Year'])['Intensity'].max().to_frame().reset_index()''', language='python')
    st.markdown('Pivot data and remove NaN values.')
    st.code('''rampen_df_intensity = rampen_df_intensity.pivot_table(index=['ISO', 'Country'], columns='Year', values='Intensity').reset_index()
rampen_df_intensity.index.name = rampen_df_intensity.columns.name = None
rampen_df_intensity = rampen_df_intensity.fillna(0)''', language='python')
    
    


if pages == 'Economic change':
    col5, col6 = st.columns([1,1])
    col7, col8 = st.columns([1,1])
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
            a = 'In ' + str(jaar) + ' in ' + landen_box + ' a disaster of type and subtype: ' + subtypen[i] + ' occured.'
            b = 'The Intensity of the disaster was: ' + str(round(intensiteit[i]*round_mult)/round_mult) + '.'
            print(a)
            with col5:
                st.markdown(a)
                st.markdown(b)

    
    grafiek_max_jaar = jaar+5
    grafiek_min_jaar = jaar-2
    grafiek_percent_jaar = jaar-3
    if grafiek_max_jaar >2021:
        grafiek_max_jaar = 2021
        
    GDP_grafiek = GDP.drop(['Country Name', 'Indicator Name', 'Indicator Code'], axis=1)
    GDP_grafiek = GDP_grafiek.set_index('Country Code').T.rename(pd.to_numeric).reset_index()
    GDP_grafiek2 = GDP_grafiek[(GDP_grafiek['index']>=grafiek_percent_jaar) & (GDP_grafiek['index']<=grafiek_max_jaar)]
    GDP_grafiek = GDP_grafiek[(GDP_grafiek['index']>=grafiek_min_jaar) & (GDP_grafiek['index']<=grafiek_max_jaar)]

    GDP_grafiek = GDP_grafiek.rename(columns={'index':'Year'}, index={'Country Code':'Index'})
    GDP_grafiek2 = GDP_grafiek2.rename(columns={'index':'Year'}, index={'Country Code':'Index'})
    GDP_land = GDP_grafiek[['Year', land_code, 'WLD']].reset_index(drop=True)
    GDP_land2 = GDP_grafiek2[['Year', land_code, 'WLD']].reset_index(drop=True)

    GDP_land['Percent'] = 0
    for index, row in GDP_land.iterrows():
        GDP_land['Percent'].iloc[index] = (GDP_land2[land_code].iloc[index+1]-GDP_land2[land_code].iloc[index])/GDP_land2[land_code].iloc[index]-\
        (GDP_land2['WLD'].iloc[index+1]-GDP_land2['WLD'].iloc[index])/GDP_land2['WLD'].iloc[index]
    GDP_land['Percent'] = GDP_land['Percent']*100    
    with col5:
        GDP_land

    GDP_fig = make_subplots(specs=[[{"secondary_y": True}]])
    GDP_fig.add_trace(
        go.Line(x=GDP_grafiek['Year'].to_list(), y=GDP_grafiek['WLD'].to_list(), name='Wereld'),
        secondary_y=True)
    GDP_fig.add_trace(
        go.Line(x=GDP_grafiek['Year'].to_list(), y=GDP_grafiek[land_code].to_list(), name=landen_box),
        secondary_y=False)
    GDP_fig.update_layout(
        title_text="<b>GDP comparison of world vs. " + landen_box +'</b>')
    GDP_fig.update_xaxes(title_text="<b>2 years before and 5 years after chosen year</b>")
    GDP_fig.update_yaxes(title_text='<b>GDP ' + landen_box + '</b>', secondary_y=False)
    GDP_fig.update_yaxes(title_text='<b>GDP world</b>', secondary_y=True)
    with col7:
        st.plotly_chart(GDP_fig)
    
   
    #scatter
    scatter_df = rampen_df[rampen_df['ISO']==land_code]
    scatter_df = scatter_df[(scatter_df['Year']>=grafiek_min_jaar) & (scatter_df['Year']<=grafiek_max_jaar)]
    scatter_graph = px.scatter(x=scatter_df['Year'], y=scatter_df['Intensity'])
    scatter_graph.update_traces(marker=dict(size=12, color='Red'))
    scatter_graph.update_layout(xaxis_range=[grafiek_min_jaar-0.25,grafiek_max_jaar+0.25],
                               yaxis_range=[-0.05,0.5])
    scatter_graph.update_xaxes(title_text="<b>2 years before and 5 years after chosen year</b>")
    scatter_graph.update_yaxes(title_text="<b>Intensity of disasters within year range</b>")
    scatter_graph.update_layout(title = 'Disaster occurences')

    with col8:
        st.plotly_chart(scatter_graph)
        
    percentage_fig = px.line(GDP_land, x='Year', y='Percent')
    percentage_fig.update_xaxes(title_text="<b>2 years before and 5 years after chosen year</b>")
    percentage_fig.update_yaxes(title_text="Percentage")
    percentage_fig.update_layout(title="<b>Difference in growth percentage of " + landen_box + ' compared to the world</b>')
    percentage_fig.add_hline(y=0)
    with col6:
        st.plotly_chart(percentage_fig)


        
    st.title('Economic change specific code')
    st.markdown('Added a page specific selectbox to the submit button code with an if statement.')
    st.code(""" if pages == 'Economic change':
    landen_naam = np.sort(rampen_df['Country'].unique())
    landen_box = st.selectbox('Choose a country', landen_naam)""", language='python')
    st.markdown('Made a dataframe to check if a disaster occurred')
    st.code('''check = rampen_df[rampen_df['Country']==landen_box]
check = check[check['Year']==jaar]''', language='python')
    st.markdown('Checked the length of the new dataframe')
    st.code("""if len(check) == 0:
    a = 'Er was geen ramp in dit jaar'
else:
    type_rampen = []
    subtypen = []
    intensiteit=[]
    for i in range(len(check)):
    type_rampen.append(check['Disaster Type'].values[i])
    subtypen.append(check['Disaster Subtype'].values[i])
    intensiteit.append(check['Intensity'].values[i])
    a = 'In ' + str(jaar) + ' in ' + landen_box + ' a disaster of type and subtype: ' + subtypen[i] + ' occured.'
    b = 'The Intensity of the disaster was: ' + str(round(intensiteit[i]*round_mult)/round_mult) + '.'""", language='python')
    st.markdown('')
    st.markdown('Graph limits were set')
    st.code('''grafiek_max_jaar = jaar+5
grafiek_min_jaar = jaar-2
grafiek_percent_jaar = jaar-3
if grafiek_max_jaar >2021:
    grafiek_max_jaar = 2021''', language='python')
    st.markdown('Graph dataframe is filtered and percentage change compared to world is calculated.')
    st.code("""GDP_grafiek = GDP.drop(['Country Name', 'Indicator Name', 'Indicator Code'], axis=1)
GDP_grafiek = GDP_grafiek.set_index('Country Code').T.rename(pd.to_numeric).reset_index()
GDP_grafiek2 = GDP_grafiek[(GDP_grafiek['index']>=grafiek_percent_jaar) & (GDP_grafiek['index']<=grafiek_max_jaar)]
GDP_grafiek = GDP_grafiek[(GDP_grafiek['index']>=grafiek_min_jaar) & (GDP_grafiek['index']<=grafiek_max_jaar)]

GDP_grafiek = GDP_grafiek.rename(columns={'index':'Year'}, index={'Country Code':'Index'})
GDP_grafiek2 = GDP_grafiek2.rename(columns={'index':'Year'}, index={'Country Code':'Index'})
GDP_land = GDP_grafiek[['Year', land_code, 'WLD']].reset_index(drop=True)
GDP_land2 = GDP_grafiek2[['Year', land_code, 'WLD']].reset_index(drop=True)

GDP_land['Percent'] = 0
for index, row in GDP_land.iterrows():
    GDP_land['Percent'].iloc[index] = (GDP_land2[land_code].iloc[index+1]-GDP_land2[land_code].iloc[index])/GDP_land2[land_code].iloc[index]-\
    (GDP_land2['WLD'].iloc[index+1]-GDP_land2['WLD'].iloc[index])/GDP_land2['WLD'].iloc[index]
GDP_land['Percent'] = GDP_land['Percent']*100""", language='python')

if pages == 'Comparison disasters': 
    col3, col4 = st.columns([1,1])
    data_subtypes = rampen_df[rampen_df['Disaster Subtype']==type_box].reset_index(drop=True)
    data_subtypes = data_subtypes.fillna(0)
    data_subtypes_jaar_0 = data_subtypes[data_subtypes['Jaar 0']!=0].sort_values(by='Category Subtypes')
    data_subtypes_jaar_1 = data_subtypes[data_subtypes['Jaar 1']!=0].sort_values(by='Category Subtypes')
    data_subtypes_jaar_2 = data_subtypes[data_subtypes['Jaar 2']!=0].sort_values(by='Category Subtypes')
    data_subtypes_jaar_3 = data_subtypes[data_subtypes['Jaar 3']!=0].sort_values(by='Category Subtypes')
    
    Fig_subtypes_jaar0 = px.box(data_subtypes_jaar_0, x='Category Subtypes', y='Jaar 0', color='Category Subtypes')
    Fig_subtypes_jaar0.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 0</b>')
    Fig_subtypes_jaar0.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar0.update_xaxes(title = '<b>Category</b>')
    
    with col3:
        st.plotly_chart(Fig_subtypes_jaar0)
    
    
    Fig_subtypes_jaar1 = px.box(data_subtypes_jaar_1, x='Category Subtypes', y='Jaar 1', color='Category Subtypes')
    Fig_subtypes_jaar1.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 1</b>')
    Fig_subtypes_jaar1.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar1.update_xaxes(title = '<b>Category</b>')
    
    with col4:
        st.plotly_chart(Fig_subtypes_jaar1)
    
    
    Fig_subtypes_jaar2 = px.box(data_subtypes_jaar_2, x='Category Subtypes', y='Jaar 2', color='Category Subtypes')
    Fig_subtypes_jaar2.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 2</b>')
    Fig_subtypes_jaar2.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar2.update_xaxes(title = '<b>Category</b>')
    
    with col3:
        st.plotly_chart(Fig_subtypes_jaar2)
    
    
    Fig_subtypes_jaar3 = px.box(data_subtypes_jaar_3, x='Category Subtypes', y='Jaar 3', color='Category Subtypes')
    Fig_subtypes_jaar3.update_layout(title='<b>'+data_subtypes['Disaster Subtype'][0] + ': Jaar 3</b>')
    Fig_subtypes_jaar3.update_yaxes(title = '<b>GDP change compared to world</b>')
    Fig_subtypes_jaar3.update_xaxes(title = '<b>Category</b>')
    
    with col4:
        st.plotly_chart(Fig_subtypes_jaar3)
    
    st.title('Comparison disasters specific code')
    st.markdown('Added a page specific selectbox to the submit button code with an if statement.')
    st.code("""if pages == 'Comparison disasters':
    types = np.sort(list(rampen_df['Disaster Subtype'].unique()))  
    type_box=st.selectbox('Choose a subtype', types)""", language='python')
    st.markdown('Selected data is filtered based on selectbox and NaN values are replaced with 0.')
    st.code("""data_subtypes = rampen_df[rampen_df['Disaster Subtype']==type_box].reset_index(drop=True)
data_subtypes = data_subtypes.fillna(0)""", language='python')
    st.markdown("Made separate dataframe based on 0's in specific columns")
    st.code("""data_subtypes_jaar_0 = data_subtypes[data_subtypes['Jaar 0']!=0].sort_values(by='Category Subtypes')
data_subtypes_jaar_1 = data_subtypes[data_subtypes['Jaar 1']!=0].sort_values(by='Category Subtypes')
data_subtypes_jaar_2 = data_subtypes[data_subtypes['Jaar 2']!=0].sort_values(by='Category Subtypes')
data_subtypes_jaar_3 = data_subtypes[data_subtypes['Jaar 3']!=0].sort_values(by='Category Subtypes')""", language='python')

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
    
    Category_data = Category_data[Category_data['Jaar 0']!=0]
    fig_a2 = px.box(Category_data, x='Disaster Type', y='Jaar 0', color='Disaster Type')
    Category_data = Category_data[Category_data['Jaar 1']!=0]
    fig_b2 = px.box(Category_data, x='Disaster Type', y='Jaar 1', color='Disaster Type')
    Category_data = Category_data[Category_data['Jaar 2']!=0]
    fig_c2 = px.box(Category_data, x='Disaster Type', y='Jaar 2', color='Disaster Type')
    Category_data = Category_data[Category_data['Jaar 3']!=0]
    fig_d2 = px.box(Category_data, x='Disaster Type', y='Jaar 3', color='Disaster Type')
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig_a2)
        st.plotly_chart(fig_c2)
    with col2:
        st.plotly_chart(fig_b2)
        st.plotly_chart(fig_d2)
        
    st.title('The Big 4 specific code')
    st.markdown('Added a page specific selectbox to the submit button code with an if statement.')
    st.code("""if pages == 'The Big 4':
    categories = ['Categorie 1', 'Categorie 2', 'Categorie 3']
    category= ['Category 1', 'Category 2', 'Category 3']
    category_dict = dict(zip(categories, category))
    category_box = st.selectbox('Choose a disaster category', categories)""", language='python')
    st.markdown('Filtered on category and disaster type and dropped rows with NaN values.')
    st.code("""Category_data = rampen_df[(rampen_df['Category Types']==1)]
Category_data = Category_data[(Category_data['Disaster Type']=='Drought') | (Category_data['Disaster Type']=='Flood') | (Category_data['Disaster Type']=='Storm') | (Category_data['Disaster Type']=='Earthquake')]
Category_data = Category_data.dropna(axis=0)
Category_data = Category_data.sort_values(by='Disaster Type').reset_index(drop=True)""", language='python')
    st.markdown('Remove rows with 0 values after making each chart')
    st.code("""Category_data = Category_data[Category_data['Jaar 0']!=0]
fig_a2 = px.box(Category_data, x='Disaster Type', y='Jaar 0', color='Disaster Type')
Category_data = Category_data[Category_data['Jaar 1']!=0]
fig_b2 = px.box(Category_data, x='Disaster Type', y='Jaar 1', color='Disaster Type')
Category_data = Category_data[Category_data['Jaar 2']!=0]
fig_c2 = px.box(Category_data, x='Disaster Type', y='Jaar 2', color='Disaster Type')
Category_data = Category_data[Category_data['Jaar 3']!=0]
fig_d2 = px.box(Category_data, x='Disaster Type', y='Jaar 3', color='Disaster Type')""", language='python')
