#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
df=pd.read_csv("/Users/vaibhavshukla/Desktop/Unemployment in India.csv",sep=",")
df=pd.read_csv("/Users/vaibhavshukla/Desktop/Unemployment_Rate_upto_11_2020.csv", sep=",")
df


# In[126]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import calendar 
import datetime as dt

import plotly.io as pio
pio.templates


# In[127]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import HTML


# In[111]:


from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


# In[36]:


df.head()


# In[16]:


df.isnull().sum()


# In[28]:


df.info()


# In[29]:


df.describe()


# In[39]:


df.columns =['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed',
             'Estimated Labour Participation Rate','Region','longitude','latitude']


# In[42]:


df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)


# In[43]:


df['Frequency']= df['Frequency'].astype('category')


# In[45]:


df['Month'] =  df['Date'].dt.month


# In[46]:


df['Month_int'] = df['Month'].apply(lambda x : int(x))


# In[47]:


df['Month_name'] =  df['Month_int'].apply(lambda x: calendar.month_abbr[x])


# In[48]:


df['Region'] = df['Region'].astype('category')


# In[49]:


df.drop(columns='Month',inplace=True)
df.head(3)


# In[50]:


df_stats = df[['Estimated Unemployment Rate',
       'Estimated Employed', 'Estimated Labour Participation Rate']]


round(df_stats.describe().T,2)


# In[51]:


region_stats = df.groupby(['Region'])[['Estimated Unemployment Rate','Estimated Employed',
                                       'Estimated Labour Participation Rate']].mean().reset_index()

region_stats = round(region_stats,2)


region_stats


# In[52]:


heat_maps = df[['Estimated Unemployment Rate',
       'Estimated Employed', 'Estimated Labour Participation Rate',
       'longitude', 'latitude', 'Month_int']]

heat_maps = heat_maps.corr()

plt.figure(figsize=(10,6))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps, annot=True,cmap='summer');


# In[112]:


fig = px.box(df,x='States',y='Estimated Unemployment Rate',color='States',title='Unemployment rate',template='plotly')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

# The below box shows unemployement rate in each state in India


# In[113]:


fig = px.scatter_matrix(df,template='plotly',
    dimensions=['Estimated Unemployment Rate','Estimated Employed',
                'Estimated Labour Participation Rate'],
    color='Region')
fig.show()


# In[103]:


plot_ump = df[['Estimated Unemployment Rate','States']]

df_unemp = plot_ump.groupby('States').mean().reset_index()

df_unemp = df_unemp.sort_values('Estimated Unemployment Rate')

fig = px.bar(df_unemp, x='States',y='Estimated Unemployment Rate',color='States',
            title='Average Unemployment Rate in each state',template='plotly')

fig.show()


# In[95]:


fig = px.bar(df, x='Region',y='Estimated Unemployment Rate',animation_frame = 'Month_name',color='States',
            title='Unemployment rate across region from Jan.2020 to Oct.2020', height=700,template='plotly')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.show()


# In[98]:


unemplo_df = df[['States','Region','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']]

unemplo = unemplo_df.groupby(['Region','States'])['Estimated Unemployment Rate'].mean().reset_index()


# In[94]:


fig = px.sunburst(unemplo, path=['Region','States'], values='Estimated Unemployment Rate',
                  color_continuous_scale='Plasma',title= 'unemployment rate in each region and state',
                  height=650,template='ggplot2')


fig.show()


# In[114]:


fig = px.scatter_geo(df,'longitude', 'latitude', color="Region",
                     hover_name="States", size="Estimated Unemployment Rate",
                     animation_frame="Month_name",scope='asia',template='plotly',title='Impack of lockdown on employement across regions')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[5,35], lonaxis_range=[65, 100],oceancolor="#6dd5ed",
    showocean=True)

fig.show()


# In[115]:


lock = df[(df['Month_int'] >= 4) & (df['Month_int'] <=7)]

bf_lock = df[(df['Month_int'] >= 1) & (df['Month_int'] <=4)]


# In[116]:


g_lock = lock.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()

g_bf_lock = bf_lock.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()


g_lock['Unemployment Rate before lockdown'] = g_bf_lock['Estimated Unemployment Rate']

g_lock.columns = ['States','Unemployment Rate after lockdown','Unemployment Rate before lockdown']

g_lock.head(2)


# In[120]:


# percentage change in unemployment rate
g_lock['percentage change in unemployment']= round(g_lock['Unemployment Rate after lockdown'] - g_lock['Unemployment Rate before lockdown']/g_lock['Unemployment Rate before lockdown'],2)


# In[121]:


plot_per = g_lock.sort_values('percentage change in unemployment')


# In[122]:


# percentage change in unemployment after lockdown

fig = px.bar(plot_per, x='States',y='percentage change in unemployment',color='percentage change in unemployment',
            title='percentage change in Unemployment in each state after lockdown',template='ggplot2')

fig.show()


# In[123]:


# function to sort value based on impact

def sort_impact(x):
    if x <= 10:
        return 'impacted States'
    elif x <= 20:
        return 'hard impacted States'
    elif x <= 30:
        return 'harder impacted States'
    elif x <= 40:
        return 'hardest impacted States'
    return x    


# In[124]:


plot_per['impact status'] = plot_per['percentage change in unemployment'].apply(lambda x:sort_impact(x))


# In[125]:


fig = px.bar(plot_per, y='States',x='percentage change in unemployment',color='impact status',
            title='Impact of lockdown on employment across states',template='ggplot2',height=650)


fig.show()


# In[ ]:




