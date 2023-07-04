#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df=pd.read_csv ("/Users/vaibhavshukla/Desktop/CarPrice_Assignment.csv", sep=",")
df


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[19]:


plt.hist(df['price'])
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Car Prices')
plt.show()


# In[44]:


plt.scatter(df['horsepower'], df['price'])
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Horsepower vs. Price')
plt.show()


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Exclude non-numeric columns from the dataframe
numeric_df = df.select_dtypes(include='number')

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})

# Rotate and align the x-axis tick labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# Rotate and align the y-axis tick labels
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right', fontsize=8)

plt.title('Correlation Matrix')
plt.show()


# In[33]:


# finding correlation of numerical and categorical features

from dython.nominal import associations

associations(df, figsize = (18, 18))
plt.show()


# In[35]:


num_cols = df.select_dtypes(exclude = 'object')
cat_cols = df.select_dtypes(include = 'object')


# In[36]:


num_cols.columns


# In[37]:


cat_cols.columns


# In[38]:


cols = num_cols.columns

plt.figure(figsize = (16, 20))
plotnumber = 1

for i in range(1, len(cols)):
    if plotnumber <= 16:
        ax = plt.subplot(4, 4, plotnumber)
        sns.histplot(x = cols[i], data = df, ax = ax, palette='rocket', kde = True, bins = 50)
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[39]:


cols = num_cols.columns

plt.figure(figsize = (16, 20))
plotnumber = 1

for i in range(1, len(cols)):
    if plotnumber <= 16:
        ax = plt.subplot(4, 4, plotnumber)
        sns.boxplot(y = cols[i], data = df, ax = ax)
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[40]:


cols = cat_cols.columns

plt.figure(figsize = (16, 20))
plotnumber = 1

for i in range(1, len(cols)):
    if plotnumber <= 10:
        ax = plt.subplot(5, 2, plotnumber)
        sns.countplot(x = cols[i], data = df, ax = ax)
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[41]:


# pie chart of fueltype column

fueltype = df['fueltype'].value_counts().reset_index()
fueltype.columns = ['fueltype', 'value_counts']
fig = px.pie(fueltype, names = 'fueltype', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[42]:


# pie chart of aspiration column

aspiration = df['aspiration'].value_counts().reset_index()
aspiration.columns = ['aspiration', 'value_counts']
fig = px.pie(aspiration, names = 'aspiration', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[45]:


# pie chart of doornumber column

doornumber = df['doornumber'].value_counts().reset_index()
doornumber.columns = ['doornumber', 'value_counts']
fig = px.pie(doornumber, names = 'doornumber', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[46]:


# pie chart of carbody column

carbody = df['carbody'].value_counts().reset_index()
carbody.columns = ['carbody', 'value_counts']
fig = px.pie(carbody, names = 'carbody', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[47]:


# pie chart of drivewheel column

drivewheel = df['drivewheel'].value_counts().reset_index()
drivewheel.columns = ['drivewheel', 'value_counts']
fig = px.pie(drivewheel, names = 'drivewheel', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[48]:


# pie chart of enginelocation column

enginelocation = df['enginelocation'].value_counts().reset_index()
enginelocation.columns = ['enginelocation', 'value_counts']
fig = px.pie(enginelocation, names = 'enginelocation', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[49]:


# pie chart of enginetype column

enginetype = df['enginetype'].value_counts().reset_index()
enginetype.columns = ['enginetype', 'value_counts']
fig = px.pie(enginetype, names = 'enginetype', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[50]:


# pie chart of cylindernumber column

cylindernumber = df['cylindernumber'].value_counts().reset_index()
cylindernumber.columns = ['cylindernumber', 'value_counts']
fig = px.pie(cylindernumber, names = 'cylindernumber', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[51]:


# pie chart of fuelsystem column

fuelsystem = df['fuelsystem'].value_counts().reset_index()
fuelsystem.columns = ['fuelsystem', 'value_counts']
fig = px.pie(fuelsystem, names = 'fuelsystem', values = 'value_counts', color_discrete_sequence = 
            px.colors.sequential.Darkmint_r, width = 650, height = 400, hole = 0.5)
fig.update_traces(textinfo = 'percent+label')


# In[52]:


#univarite and bi-variates  analysis
cols = num_cols.columns

plt.figure(figsize = (16, 20))
plotnumber = 1

# plotting the countplot of each categorical column.

for i in range(1, len(cols)):
    if plotnumber <= 16:
        ax = plt.subplot(4, 4, plotnumber)
        sns.scatterplot(x = cols[i], y = df['price'], data = df, ax = ax, palette='rocket')
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[53]:


cols = cat_cols.columns

plt.figure(figsize = (16, 25))
plotnumber = 1

for i in range(1, len(cols)):
    if plotnumber <= 10:
        ax = plt.subplot(5, 2, plotnumber)
        sns.boxplot(x = cols[i] ,y = df['price'], data = df, ax = ax)
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[54]:


df['CarName'].value_counts()


# In[55]:


# extracting companies names 

df['CarName'] = df['CarName'].str.split(' ', expand = True)[0]
df['CarName'].value_counts()


# In[56]:


# handling duplicate values 

df['CarName'] = df['CarName'].replace({'toyouta': 'toyota', 'Nissan': 'nissan', 'maxda': 'mazda', 'vokswagen': 'volkswagen',
                                      'vw': 'volkswagen', 'porcshce': 'porsche'})
df['CarName'].value_counts()


# In[57]:


df1 = pd.DataFrame(df['CarName'].value_counts().reset_index())
df1.columns = ['Car Name', 'No of Cars']

px.bar(data_frame = df1, x = 'No of Cars', y = 'Car Name', color = 'No of Cars', template = 'ggplot2')


# In[58]:


px.scatter(data_frame = df, x = 'carlength', y = 'price', color = 'doornumber')


# In[59]:


px.scatter(data_frame = df, x = 'carlength', y = 'price', color = 'carbody')


# In[60]:


px.scatter(data_frame = df, x = 'carwidth', y = 'price', color = 'carbody')


# In[61]:


px.scatter(data_frame = df, x = 'enginesize', y = 'price', color = 'carbody')


# In[62]:


px.scatter(data_frame = df, x = 'horsepower', y = 'price', color = 'carbody')


# In[63]:


px.scatter(data_frame = df, x = 'enginesize', y = 'price', color = 'enginetype')


# In[64]:


px.scatter(data_frame = df, x = 'horsepower', y = 'price', color = 'enginetype')


# In[65]:


px.scatter(data_frame = df, x = 'citympg', y = 'price', color = 'enginetype')


# In[66]:


px.scatter(data_frame = df, x = 'highwaympg', y = 'price', color = 'enginetype')


# In[67]:


px.scatter(data_frame = df, x = 'citympg', y = 'price', color = 'fuelsystem')


# In[68]:


px.scatter(data_frame = df, x = 'highwaympg', y = 'price', color = 'fuelsystem')


# In[69]:


px.histogram(data_frame = df, x = 'wheelbase', nbins = 50, color = 'aspiration', template = 'ggplot2',
             marginal = 'box', barmode = 'group')


# In[70]:


px.scatter(data_frame = df, x = 'wheelbase', y = 'price', color = 'aspiration', template = 'ggplot2')


# In[71]:


px.histogram(data_frame = df, x = 'wheelbase', color = 'carbody', template = 'ggplot2',
             marginal = 'box', barmode = 'group', height = 600)


# In[72]:


px.scatter(data_frame = df, x = 'wheelbase', y = 'price', color = 'carbody', template = 'ggplot2')


# In[73]:


px.histogram(data_frame = df, x = 'wheelbase', nbins = 50, color = 'drivewheel', template = 'ggplot2',
             marginal = 'box', barmode = 'group')


# In[74]:


px.scatter(data_frame = df, x = 'wheelbase', y = 'price', color = 'drivewheel', template = 'ggplot2')


# In[75]:


px.histogram(data_frame = df, x = 'curbweight', nbins =100,color = 'aspiration', template = 'ggplot2',
             marginal = 'box', barmode = 'group')


# In[76]:


px.scatter(data_frame = df, x = 'curbweight', y = 'price',color = 'aspiration', template = 'ggplot2')


# In[77]:


px.histogram(data_frame = df, x = 'curbweight', nbins = 50, color = 'carbody', template = 'ggplot2',
             marginal = 'box', height = 725, barmode = 'group')


# In[78]:


px.scatter(data_frame = df, x = 'curbweight', y = 'price', color = 'carbody', template = 'ggplot2')


# In[79]:


px.histogram(data_frame = df, x = 'curbweight', nbins = 50, color = 'drivewheel', template = 'ggplot2',
             marginal = 'box', barmode = 'group')


# In[80]:


px.scatter(data_frame = df, x = 'curbweight', y = 'price', color = 'drivewheel', template = 'ggplot2')


# In[81]:


px.histogram(data_frame = df, x = 'curbweight', nbins = 100, color = 'enginelocation', template = 'ggplot2',
             marginal = 'box', barmode = 'group')


# In[82]:


px.scatter(data_frame = df, x = 'curbweight', y = 'price', color = 'enginelocation', template = 'ggplot2')


# In[83]:


px.scatter(data_frame = df, x = 'curbweight', y = 'price',color = 'fuelsystem', template = 'ggplot2')


# In[84]:


#data pre-processing 
df.head()


# In[85]:


df.drop(columns = ['car_ID'], axis = 1, inplace = True)
df.head()


# In[86]:


cat_cols = df.select_dtypes(include = 'object')
cat_cols.head()


# In[87]:


df['cylindernumber'].value_counts()


# In[88]:


# checking for outliers

cols = num_cols.columns

plt.figure(figsize = (16, 20))
plotnumber = 1

# plotting the countplot of each categorical column.

for i in range(1, len(cols)):
    if plotnumber <= 16:
        ax = plt.subplot(4, 4, plotnumber)
        sns.boxplot(x = cols[i], data = df, ax = ax, palette='rocket')
        plt.title(f"\n{cols[i]} \n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[89]:


# encoding ordinal categorical columns

df['doornumber'] = df['doornumber'].map({'two': 2, 'four': 4})
df['cylindernumber'] = df['cylindernumber'].map({'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})


# In[91]:


df.head()


# In[93]:


df.shape


# In[97]:


columns_to_drop = ['fueltype', 'carbody', 'aspiration', 'symboling', 'wheelbase', 'cylindernumber', 'doornumber', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'enginelocation']

# Check if the columns exist in the DataFrame before dropping
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Drop the existing columns from the DataFrame
df.drop(columns=columns_to_drop_existing, inplace=True)



# In[99]:


df.shape


# In[100]:


df.head()


# In[101]:


# creating features and label variable

X = df.drop(columns = 'price', axis = 1)
y = df['price']


# In[102]:


X = pd.get_dummies(X, drop_first = True)
X.head()


# In[103]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[104]:


# scaling data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


# In[105]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[106]:


lr.score(X_train, y_train)


# In[107]:


lr.score(X_test, y_test)


# In[112]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[113]:


dtr.score(X_train, y_train)


# In[114]:


dtr.score(X_test, y_test)


# In[115]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[116]:


rfr.score(X_train, y_train)


# In[117]:


rfr.score(X_test, y_test)


# In[118]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)


# In[119]:


gbr.score(X_train, y_train)


# In[120]:


gbr.score(X_test, y_test)


# In[123]:


models = pd.DataFrame({
    'Model' : ['Linear Regression', 'Decision Tree', 'Random Forest','Gradient Boost',],
    'Score' : [lr.score(X_test, y_test), dtr.score(X_test, y_test), rfr.score(X_test, y_test),
               gbr.score(X_test, y_test),]
})

models.sort_values(by = 'Score', ascending = False)


# In[124]:


px.bar(data_frame = models, x = 'Score', y = 'Model', 
       color = 'Score', template = 'plotly_dark', title = 'Models Comparison')


# In[ ]:




