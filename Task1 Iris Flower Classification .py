#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("/Users/vaibhavshukla/Desktop/Iris.csv",sep=",")
df


# In[2]:


import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('Iris.csv')
df.head()


# In[4]:


#delete a cloumn
df = df.drop(columns = ['Id'])
df.head()


# In[5]:


#to display stats about data
df.describe()


# In[8]:


df.info()


# In[9]:


df['Species'].value_counts()


# In[10]:


df.isnull().sum()


# In[115]:


co=x.corr()
co


# In[116]:


data = np.random.randint(low = 1,
                         high = 100,
                         size = (10, 10))


# In[117]:


sns.set(rc = {'figure.figsize':(10,6)})
sns.heatmap(data=co,annot=True)
plt.show()


# In[18]:


sns.pairplot(df, hue="Species")
plt.legend 


# In[61]:


# Separate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]
print(Y)


# In[128]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print (X_train)


# In[129]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)


# In[130]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x_train,y_train)


# In[132]:


predictions = classifier.predict(x_test)


# In[133]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[134]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[88]:


#logistics Regression 
from sklearn.linear_model import LogisticRegression 
model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)


# In[139]:


m=model.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(x_test, y_test)))


# In[143]:


model.score(x,y)


# In[112]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[50]:


# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)
# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(X_new)


# In[ ]:




