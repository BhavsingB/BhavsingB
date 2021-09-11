#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification
# Decision tree is nothing but Taking decision , and Is a kind of tree structure 
# 

# #### Entropy is nothing but is a mesurement of pure data 
# Entropy range is 0 to 1
# 
# Entropy is 0 === is totaly pure data
# Entropy is 1 === is impure data
# 
# from information theory 
# 
# Information Gain
# is nothing but respectiuly data,
# information entropy befor spliting 
# information entropy after spliting 
# 
# which will get less entropy it will be a Root Node

# #### Import The Libraris 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pwd


# #### Load the dataset

# In[3]:


df=pd.read_csv(r'C://Users//HOME/Social_Network_Ads.csv')


# #### Understand the dataset

# In[4]:


# data information and the size 
df.info()


# In[5]:


# check the datatype
df.dtypes


# In[6]:


df.head() # top 5 rows


# In[7]:


df.tail() # bottom 5 rows


# ### Handle the NaN Values

# In[8]:


# check the NaN values
df.isnull().sum()


# # Handle the Outliers 

# In[9]:


df.boxplot()


# In[10]:


sns.boxplot(df['EstimatedSalary'])


# In[11]:


sns.scatterplot(df['Age'],df['EstimatedSalary'],hue=df['Purchased'])


# In[12]:


df.describe()


# ## Split X and Y

# In[13]:


x=df.iloc[:,[2,3]]
x 


# In[14]:


y=df.iloc[:,4]
y


# ## Split Train and test 

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
# what is Random_state 
# what will hapend without Random_state, when will run multiple time every time it represent deferent values.
# once will use Random_state and we can run multiple time it can be represent same values


# In[17]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Decision Tree no need a Scaling part 

# In[18]:


#from sklearn.preprocessing import MinMaxScaler


# In[19]:


#sc=MinMaxScaler()


# In[20]:


#x_train=sc.fit_transform(x_train)


# In[21]:


#x_test=sc.transform(x_test)


# # Build Model Decision Tree Classification 

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# ### what are the Hyeparameters are in main here 

# In[32]:


# creaing the instance of model
# using parametr criterion="entropy" to calculate the Information gain
# another formal also have is Gini as a same like Entropy


# dt_model=DecisionTreeClassifier(criterion="entropy")
dt_model=DecisionTreeClassifier(criterion="gini",max_depth=4)


# In[33]:


# Train the model
dt_model.fit(x_train, y_train)


# ## Prediction

# #### Predict for Test 

# In[34]:


y_pred=dt_model.predict(x_test) # by defualt thrisold value given 0 to 1


# In[35]:


# orginal prabpliy value
dt_model.predict(x_test)
# by left side of prab 0, by left side prab 1 


# In[36]:


y_pred


# #### Predict 
# for Train but not Necessary to predict to Train 

# In[37]:


y_pred_train=dt_model.predict(x_train)


# # Evaluation

# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


print("Test Accuracy :::",accuracy_score(y_test, y_pred)*100,"%")
print("Train Accuracy :::",accuracy_score(y_train, y_pred_train)*100,"%")


# ### Is a kind of overfitting problem
# Model is good for train data but not goot for test data
# 
# The Solution 
# 1. Regularization  (In decision tree not have Regularization
# 2. Hyperparameter Tuning
# 
# Let's go with Hyperparameter Tuning to solve the Over fitting problem
# 

# ### Confusion Metrix

# In[40]:


from sklearn.metrics import confusion_matrix

# Making the confusion matrix 
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix


# In[41]:


confusion_matrix(y_test,y_pred)


# In[42]:


print(metrics.accuracy_score(y_test, y_pred)) # to find accuracy score of confution metrix


# In[43]:


plot_confusion_matrix(confusion_matrix(y_test,y_pred),class_names=["Not Purchased", "Purchased"])


# In[44]:


# recall precision f1score
from sklearn.metrics import classification_report


# In[45]:


print(classification_report(y_test, y_pred))


# In[46]:


# Class zero is some good 


# ## Plot Decision Boundry

# In[47]:


from mlxtend.plotting import plot_decision_regions,plot_confusion_matrix


# In[48]:


x_train.shape


# In[49]:


x_train.values


# In[50]:


y_train.values


# In[51]:


type(y_train.values)


# In[52]:


plot_decision_regions(x_train,y_train.values,clf=dt_model)


# In[ ]:


plot_decision_regions(x, y, clf=dt_model, legend=2)


# ## plot the Decision Tree

# In[53]:


from sklearn.tree import plot_tree


# In[54]:


plot_tree(dt_model)


# In[55]:


plt.figure(figsize=(15,15))
plot_tree(dt_model,fontsize=9,filled=True,feature_names=['Age','Salary']) # for color
plt.show() # to remove all the txt
# for color

As per this graph
x[0]<=44.5
entropy=0.942 
samples=320 
value=[205,115]

# depend on sample 
# have least entropy that's why it will be a Root node
# In[56]:


# for more understand
x_train[x_train['Age']<=44.5].shape


# In[57]:


x_train[x_train['Age']>44.5].shape


# # Real time prediction

# In[58]:


age=25
salary = 80000


# In[59]:


data=np.array([[age,salary]])


# In[60]:


data


# In[61]:


dt_model.predict(data)


# # Decision tree Regression problem
# how to work 
# 
# profit
# 40
# 60 
# 60+40 /2=50
# y_pred=50 # actual data point (avarage also)
# sum(y-y^) ===10+10=20
# 
# To find for give value like less then and greterthen to fine deviation
# which will get least Entropy that will be Root
# 
# For Example
# Decision Tree output is avarage the Regian
# R&D Spend<100 --120
# R&D Spend<400 --180
# R&D Spend<500 --160
# R&D Spend<700 --110
# R&D Spend<800 --150
# 
# like this calculate going untill the deviation will be 0 and this is a best split and Root node

# In[ ]:


Classification -- Entropy or (Gini)
Regression -- Deviation or Error

