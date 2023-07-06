#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv('/Users/vaibhavshukla/Desktop/spam.csv', encoding='latin-1')
df


# In[7]:


df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.isnull().sum()


# In[11]:


df.info


# In[12]:


x=df.v2
x


# In[13]:


y.replace(to_replace='ham',value=1,inplace=True)
y.replace(to_replace='spam',value=0,inplace=True)
y


# In[14]:


y=df.v1
y.value_counts()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df'

# Create a bar plot of the class distribution
class_counts = df['v1'].value_counts()
class_counts.plot(kind='bar')
plt.title('Class Distribution of Spam/Ham')
plt.xlabel('Spam/Ham')
plt.ylabel('Number of Mails')
plt.show()


# In[17]:


df.describe()


# In[28]:


# Preprocess the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=25)


# In[3]:


from sklearn.naive_bayes import MultinomialNB

# Create a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train, y_train)


# In[4]:


from sklearn.metrics import classification_report

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[5]:


# Example usage of the trained model for spam detection
new_email_text = ["Get a free vacation now!"]
new_email_text_transformed = vectorizer.transform(new_email_text)
prediction = classifier.predict(new_email_text_transformed)
print("Prediction:", prediction)


# In[ ]:




