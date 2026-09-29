#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np 


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn.linear_model import LogisticRegression


# In[5]:


from sklearn.preprocessing import LabelEncoder ,Binarizer,MinMaxScaler,StandardScaler


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


from sklearn.svm import SVC


# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


Titanic=pd.read_csv("Titanic-Dataset.csv")


# In[11]:


Titanic


# In[12]:


Titanic.shape


# In[13]:


Titanic.describe()


# In[14]:


Titanic.info()


# In[15]:


Titanic.isnull().sum()


# In[16]:


Titanic["Age"]=Titanic["Age"].fillna(Titanic["Age"].median())


# In[17]:


Titanic["Cabin"]=Titanic["Cabin"].fillna(Titanic["Cabin"].mode()[0])


# In[18]:


Titanic["Embarked"]=Titanic["Embarked"].fillna(Titanic["Embarked"].mode()[0])


# In[19]:


Titanic.isnull().sum()


# In[20]:


Titanic["Cabin"].unique


# In[21]:


Titanic["Ticket"].unique


# In[22]:


Titanic["Embarked"].unique


# In[23]:


Titanic["Cabin"].value_counts()


# In[24]:


Titanic["Fare"].value_counts()


# In[25]:


Titanic["Ticket"].value_counts()


# In[26]:


Titanic["Age"].value_counts()


# In[27]:


Titanic["Embarked"].value_counts()


# In[28]:


Titanic=Titanic.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[29]:


Titanic.head()


# In[30]:


Titanic["Survived"].value_counts()


# In[31]:


Titanic["Survived"].value_counts().plot(kind="bar")


# In[32]:


s=Titanic.groupby("Sex")["Survived"].mean()


# In[33]:


pc=Titanic.groupby("Pclass")["Survived"].mean()



# In[34]:


s.plot(kind="bar")


# In[35]:


pc.plot(kind="bar")


# In[ ]:





# In[36]:


label=LabelEncoder()


# In[37]:


Titanic["Sex"]=label.fit_transform(Titanic["Sex"])


# In[38]:


Titanic["Embarked"]=label.fit_transform(Titanic["Embarked"])


# In[39]:


Titanic.head()


# In[40]:


binarizer=Binarizer(threshold=20.0000)


# In[41]:


Titanic["Fare"]=binarizer.fit_transform(Titanic[["Fare"]])


# In[42]:


Titanic.head()


# In[43]:


x=Titanic.drop("Survived",axis=1)
y=Titanic["Survived"]


# In[44]:


x.shape


# In[45]:


y.shape


# In[46]:


x


# In[47]:


y


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(
x,
y,
test_size=0.20,
random_state=42)


# In[49]:


x_train.shape


# In[50]:


x_test.shape


# In[51]:


Y=Titanic[["Age"]]


# In[52]:


model=LogisticRegression()


# In[53]:


model.fit(x_train,y_train)


# In[54]:


predict= model.predict(x_test)


# In[55]:


predict[4]


# In[56]:


predict[39]


# In[57]:


svm= SVC(kernel="rbf")


# In[58]:


svm.fit(x_train,y_train)


# In[59]:


predict= svm.predict(x)


# In[60]:


predict[38]


# In[61]:


predict[:20]


# In[62]:


x


# In[63]:


KNN= KNeighborsClassifier()


# In[64]:


KNN.fit(x_train,y_train)


# In[65]:


KNN_predict=KNN.predict(x_train)


# In[66]:


KNN_predict[:20]


# In[67]:


KNN_predict[177]


# In[68]:


scaler= StandardScaler()


# In[69]:


x_train_scaled=scaler.fit_transform(x_train)


# In[70]:


x_test_scaled=scaler.fit_transform(x_test)


# In[ ]:





# In[71]:


from sklearn.metrics import (     accuracy_score,     precision_score,     recall_score,     f1_score ) 


# In[72]:


def evaluate_model(name, y_test, y_pred):   
    print("\n", name)     
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print("Precision:",precision_score(y_test, y_pred))
    print("Recall:",recall_score(y_test, y_pred))      
    print("F1 Score:",f1_score(y_test, y_pred))


# In[73]:


import joblib

joblib.dump(
    svm,
    "titanic_svm_model.pkl"
)

joblib.dump(
    scaler,
    "titanic_scaler.pkl"
)

print(
    "\nModel and scaler saved successfully!"
)



# In[ ]:




