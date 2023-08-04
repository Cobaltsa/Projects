#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import geopandas as gpd
import sklearn as sk


dfs=pd.read_csv("/Users/venmalladi/Downloads/COVID-19_Cases.csv",sep=",",iterator=True,chunksize = 1000000)
COVID_DATA=pd.concat(dfs)
POP_DATA=pd.read_csv("/Users/venmalladi/Downloads/Book9.csv")
STATE_ABBREV=pd.read_csv("/Users/venmalladi/Downloads/csvData-2.csv")
#COVID_DATA=pd.read_csv("/Users/venmalladi/Downloads/Book3.csv") 
    
          
          
         


# In[2]:


import numpy as np


# In[ ]:





# In[3]:


val1 = input("Enter the Location: ")
print(val1)
val2 = int(input("Enter the Age: "))
print(val2)
val3 = input("Enter the Date: ")
print(val3)


# In[ ]:


#likelihood
#Pseudo Code
#If age is between X and Y
#count age between X and Y
    #if location is Z
    #count location
        #if date is in this range 
        #count location
        #print count/population in location
#for sensitivity of dates
    # if age is between X and Y
        #print count of cases between the two values


# In[1]:


#Graph Logic
#if age is between this date and another date
    #count number of cases and deaths within the age group
    #highlight whole map and list number of cases and deaths
#if location is = to one of the locations
    #count number of cases and deaths within that location
    #highlight the regions where these cases are within these locations
    


# In[3]:


#hierarchical clustering of locations
#Based on #of Cases and County x = number of cases in each county y = county 
#Calculate the Pearsons Coefficient(comparing every county to each other)
#Use Complete Linkage Hierarchical Clustering Algorthim for this one (# of clusters variable parameters that I can change)
#https://plotly.com/python/county-choropleth/
#Figure out how to plot these graphs in reality


# In[7]:


#Likelihood calculation
#for val1 in dataframe if val1 = state value in data frame
#print the count of how many cases were in those states
#for those values, find the age group and print those counts


# In[ ]:





# In[18]:


#This piece of code gives you how many cases with the given age and location the user provided & infection rate
int(val2)

#Data Cleaning
import numpy as np

COVID_DATA["case_month"]=COVID_DATA["case_month"].apply(str)
COVID_DATA['case_month'] = COVID_DATA['case_month'].str.replace('-', '')
COVID_DATA['age_group'] = COVID_DATA['age_group'].str.replace(' ', '')
val3=val3.replace("-", "")
#fipscode_str=COVID_DATA['county_fips_code'].apply(lambda x: '{0:0>5}'.format(x))
#COVID_DATA['county_fips_code']=fipscode_str.astype(int())
#COVID_DATA['county_fips_code'] = fipscode_str.astype(int())
COVID_DATA['res_state'] = COVID_DATA['res_state'].astype(str)
COVID_DATA['Year_Val'] = COVID_DATA['case_month'].astype(str).str[:4]

#IF then loop based on use inputs
if (0 < val2 <= 17):
    age_group1 = "0-17years"
    Y = COVID_DATA[(COVID_DATA['res_state']==val1) & (COVID_DATA['age_group']==age_group1) & (COVID_DATA['case_month'] <=val3) & (COVID_DATA['Year_Val'])]
    filtered_cases = Y.shape[0]
    first_value = Y['Year_Val'].iat[0]
    first_value=int(first_value)
    res = POP_DATA.dtypes
    Z = POP_DATA[(POP_DATA['Age_Group']=="0-17years") & (POP_DATA['Year'] == first_value) & (POP_DATA['res_state']==val1)]
    YZ = POP_DATA[(POP_DATA['Age_Group']=="0-17years") & (POP_DATA['Year'] == first_value)]
    cases_age_group = Z['Population'].sum()
    cases_age_group2 = YZ['Population'].sum()
    infection_rate = filtered_cases/cases_age_group 
    infection_rate2 = filtered_cases/cases_age_group2
    Cases= Y['case_month'].count()
    Y['Cases'] = Cases
    Y['Infection Rate State'] = infection_rate
    Y['Infection Rate US'] =infection_rate2
elif 18 <= val2 <= 64:
    age_group2 = "18to49years"
    age_group3 = "50to64years"
    X = COVID_DATA[(COVID_DATA['res_state']==val1) & (COVID_DATA['age_group']==age_group2) & (COVID_DATA['case_month'] <=val3)]
    YY = COVID_DATA[(COVID_DATA['res_state']==val1) & (COVID_DATA['age_group']==age_group3) & (COVID_DATA['case_month'] <=val3)]
    numb_inf_age1 = X.shape[0]
    numb_inf_age2 = YY.shape[0]
    filtered_cases = numb_inf_age1+numb_inf_age2                                               
    first_value = X['Year_Val'].iat[0]
    first_value=int(first_value)
    print(type(first_value))
    res = POP_DATA.dtypes
    Z = POP_DATA[(POP_DATA['Age_Group']=="18-65years") & (POP_DATA['Year'] == first_value) & (POP_DATA['res_state']==val1)]
    YZ = POP_DATA[(POP_DATA['Age_Group']=="18-65years") & (POP_DATA['Year'] == first_value)]
    cases_age_group = Z['Population'].sum()
    cases_age_group2 = YZ['Population'].sum()
    print(cases_age_group)
    print(cases_age_group2)
    infection_rate = filtered_cases/cases_age_group 
    infection_rate2 = filtered_cases/cases_age_group2
    print(infection_rate)
    print(infection_rate2)
    #Combining the two datasets
    Combination = [X,YY]
    Y = pd.concat(Combination)
    Cases= Y['case_month'].count()
    print(Cases)
    Y['Cases'] = Cases
    Y['Infection Rate State'] = infection_rate
    Y['Infection Rate US'] =infection_rate2
elif val2 > 64:
    age_group4 = "65+years"
    Y = COVID_DATA[(COVID_DATA['res_state']==val1) & (COVID_DATA['age_group']==age_group4) & (COVID_DATA['case_month'] <=val3) & (COVID_DATA['Year_Val'])]
    filtered_cases = Y.shape[0]
    first_value = Y['Year_Val'].iat[0]
    first_value=int(first_value)
    print(type(first_value))
    res = POP_DATA.dtypes
    print(res)
    Z = POP_DATA[(POP_DATA['Age_Group']=="65+years") & (POP_DATA['Year'] == first_value) & (POP_DATA['res_state']==val1)]
    YZ = POP_DATA[(POP_DATA['Age_Group']=="65+years") & (POP_DATA['Year'] == first_value)]
    print(Z)
    cases_age_group = Z['Population'].sum()
    cases_age_group2 = YZ['Population'].sum()
    print(cases_age_group)
    print(cases_age_group2)
    infection_rate = filtered_cases/cases_age_group 
    infection_rate2 = filtered_cases/cases_age_group2
    print(infection_rate)
    print(infection_rate2)
    Cases= Y['case_month'].count()
    print(Cases)
    Y['Cases'] = Cases
    Y['Infection Rate State'] = infection_rate
    Y['Infection Rate US'] =infection_rate2


# In[8]:


#COVID_DATA['county_fips_code'] = fipscode_str.astype('int')


# In[ ]:





# In[5]:


#US map generation
#Need to figure out how to process so much data

from plotly.tools import FigureFactory as ff
from plotly.figure_factory._county_choropleth import create_choropleth
import warnings
from shapely.errors import ShapelyDeprecationWarning
import plotly.express as px
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import plotly.graph_objects as go


# In[19]:


import plotly.express as px

#fig = px.choropleth(Y,
#                    locations='res_state', 
#                    locationmode="USA-states", 
#                    scope="usa",
#                    color='Cases',
#                    color_continuous_scale="Viridis_r", 
#                    
#                    )
#fig.show()

import plotly.graph_objects as go

for col in Y.columns:
    Y[col] = Y[col].astype(str)
    
Y['text'] = Y['res_state'] + '<br>' +     ' Infection Rate State: ' + Y['Infection Rate State'] + '<br>' +     ' Infection Rate US: ' + Y['Infection Rate US'] + '<br>' +     ' Number of Cases: ' + Y['Cases']


fig = go.Figure(data=go.Choropleth(
    locations=Y['res_state'],
    z=Y['Cases'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=Y['text'], # hover text
    marker_line_color='black', # line markers between states
    colorbar_title="Number of Cases"
))

fig.update_layout(
    title_text='COVID-19 Cases and Infection Rate',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()


# In[29]:


#This is an example I am testing and then trying to apply this to my work

#Hierarchical Data Clustering
#Compute the proximity matrix
#Let each data point be a cluster
#Repeat: Merge the two closest clusters and update the proximity matrix
#Until only a single cluster remains

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

METHOD_DATA = COVID_DATA.iloc[:, [0, 2]].values

dendrogram = sch.dendrogram(sch.linkage(METHOD_DATA, method='complete'))

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
model.fit(METHOD_DATA)
labels = model.labels_

#plt.scatter(METHOD_DATA[labels==0, 0], METHOD_DATA[labels==0, 1], s=50, marker='o', color='red')
#plt.scatter(METHOD_DATA[labels==1, 0], METHOD_DATA[labels==1, 1], s=50, marker='o', color='blue')
#plt.scatter(METHOD_DATA[labels==2, 0], METHOD_DATA[labels==2, 1], s=50, marker='o', color='green')
#plt.scatter(METHOD_DATA[labels==3, 0], METHOD_DATA[labels==3, 1], s=50, marker='o', color='purple')
#plt.scatter(METHOD_DATA[labels==4, 0], METHOD_DATA[labels==4, 1], s=50, marker='o', color='orange')
#plt.show()


# In[7]:


county_data = {}

import math

for i, j in COVID_DATA.iterrows():
    if not math.isnan(j['county_fips_code']):
        fips_code = int(j['county_fips_code'])
        fips_count = 0
        if fips_code not in county_data:
            county_data[fips_code] = 1
        else:
            updated_count = county_data.get(fips_code)+1
            temp = {fips_code: updated_count}
            county_data.update(temp)

print(county_data)    
#correlation of time series between two different regions
#dis

#dictionary fips code and a count and a bolleeen dictionary of the county_fips_counts of dictionaries
#You will have a count that start at 0, bolleeen

#import numpy as np

#COVID_DATA['case_month'] = COVID_DATA['case_month'].astype(int)

#temp=[]

#corr = np.corrcoef(list(county_data.keys()), list(county_data.values()))

#print(corr)
#distance matrix plug into orange canvas and use that directly?
#ex: 1000 counties, 1000x1000 matrix correlation between county i and county j
#pandas corr function
#pandas dataframe that looks at each county, each column is diff county,
#corr -> correlation coefficient
#county fips -> integer

#case month should be index
#fips code is a column
#aggegate data to create plot


# In[9]:


from sklearn.metrics.pairwise import check_pairwise_arrays

COVID_DATA_Agg = pd.pivot_table(COVID_DATA,
               values='res_county',
               index = 'case_month',
               columns = 'county_fips_code',
               aggfunc='count')

COVID_DATA_Agg = COVID_DATA_Agg.fillna(0)
print(COVID_DATA_Agg)
corr = np.corrcoef(COVID_DATA_Agg,rowvar=False)

corr= np.where(np.isfinite(corr), corr, 0)

corr.shape
#np.any(np.isnan(corr))
#np.all(np.isfinite(corr))


# In[14]:


#Using the correlation Matrix calculated 

#Hierarchical Data Clustering
#Compute the proximity matrix
#Let each data point be a cluster
#Repeat: Merge the two closest clusters and update the proximity matrix
#Until only a single cluster remains

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

#COVID_DATA['res_state'] = COVID_DATA['res_state'].astype()

#METHOD_DATA1 = [(x,y) for x,y in zip(list(county_data.keys()),list(county_data.values()))]

#METHOD_DATA2 = pd.DataFrame(METHOD_DATA1, columns=['fips_code', 'fips_count'])

#print(METHOD_DATA2)

#METHOD_DATA = METHOD_DATA2.iloc[:, [0, 1]].values

#print(METHOD_DATA)

#dendrogram = sch.dendrogram(sch.linkage(METHOD_DATA, method='complete'))

#model = AgglomerativeClustering(affinity=corr, n_clusters=2, linkage='complete')
#print(model.labels_)
num_clusters = 5

model = AgglomerativeClustering(n_clusters = num_clusters, affinity="precomputed", linkage='complete',compute_distances = True)
model.fit(corr)
labels = model.labels_

from plotly.tools import FigureFactory as ff
from plotly.figure_factory._county_choropleth import create_choropleth
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

fips = [str(x) for x in list(county_data.keys())]
total_cases = [str(x) for x in list(county_data.values())]
labels = [str(x) for x in labels]
values = labels

#print(labels)
#fig = create_choropleth(fips=fips, values=values)

#fig.layout.template = None
#fig.show()

print(len(values))

#multi-threading for reading all the data
#simaphore?
#count on what the next set of data will be
#workers for multi-thread off the number - critcal section off of it
#read in the value, then increment by 1
#method for function for updating the data
#workers * specific to threading
#reading data in + writing data in
#read in the data structure

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
#df = [fips][values][# of cases]
print(fipscode_str)
zipData = zip(fipscode_str.values.tolist(),values,county_data.values())

df = pd.DataFrame (zipData, columns = ['fips','cluster','total_cases'])   
print(df)    
import plotly.express as px

fig = px.choropleth(df, geojson=counties, locations='fips', color='cluster',hover_name='total_cases',
                           color_continuous_scale="Viridis",
                           range_color=(0, num_clusters),
                           scope="usa",
                           labels={'unemp':'unemployment rate'}
                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#fips   #values - 0,1,2,3 - clusters


# In[13]:


fipscode_str=COVID_DATA['county_fips_code'].apply(lambda x: '{0:0>5}'.format(x))
COVID_DATA['county_fips_code']=fipscode_str.astype(int)


# In[ ]:




