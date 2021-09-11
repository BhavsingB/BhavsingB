#!/usr/bin/env python
# coding: utf-8

# # Decision Tree 
# Decision Tree is a CART
# 
# 1. CART (Classification And Regression Tree)
# 
# Decision tree is nothing but Taking decision
# 1. Deside to either will go for class or not
# 2. Deside to either will go for movie or not
# etc., and it Is a kind of tree structure format.
# 
# Decision tree is a Information based
# 
# ### Calculate
# Entropy is formula to Calculate Gini also smae,
# Entropy is nothing but is a mesurement of pure data 
# 
# Entropy range is 0 to 1
# 
# 1. Entropy is 0 === is totaly pure data
# 2. Entropy is 1 === is impure data
# 
# from information theory 
# 
# #### Information Gain
# 1. is nothing but respectiuly data,
# 2. information entropy befor spliting 
# 3. information entropy after spliting 
# 
# which will get less entropy it will be a Root Node
# 
# ### in Regression split untill mse will 0  

# #### Decision Tree
# Model of Hyperparameter Tunning 
# 
# Linear Regression No
# 
# Logistic Regression Have a 1 Hyperparameter
# 
# 

# # Decision Tree Regression 
# Classification need a kind of classifire data like yes, no, 1,0,good,bad
# Regression need a kind of Regresire data like profit, value,
# 
# let see how to calculate Entropy in Numerical data

# ### For example will take age
# 
# age=([20,20,50,20,50,50])
# Purchase=([0,1,0,0,1,0])
# -(3/6*math.log2(3/6)+3/6*math.log2(3/6))
# 
# age<=20 
# age>20 
# calculate like that 
# 
# 
# which ever age is have least entropy that will be finly root node

# # Decision Tree Regression 

# #### import nacessary libraryis 

# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[94]:


pwd


# #### Load the dataset

# In[95]:


df=pd.read_csv(r'C://Users//HOME/50_Startups.csv')


# #### Understand the dataset

# In[96]:


# data information and the size 
df.info()


# In[97]:


df.columns # check the columns


# In[98]:


df.isnull().sum()


# In[99]:


# find the mean value
df.mean()


# In[100]:


# fill nan with mean
df.fillna(df.mean(),inplace=True)


# In[101]:


# find the missing value 
df.isnull().sum()


# #### In numeric format data Will fillna with mean value lets look into the state column it is a categorical format

# #### Still have nan for State, lets see proper what are values in state

# In[102]:


# find gender
df["State"].value_counts()


# In[103]:


sns.countplot(df["State"])


# In[104]:


# Compere Gender to Purchased who will purchased high
pd.crosstab(df["State"],df["Profit"])


# #### Lets fillna with mode 

# #### Lets fillna with mode 

# In[105]:


# find the mode value which is the first
df['State'].mode()[0]


# In[106]:


# lets fill nan with 0
df['State'].fillna(df['State'].mode()[0],inplace=True)


# In[107]:


# check the nan again it was done ar not
df.isnull().sum()


# ### Done filling nan

# ### Handling the Outliers  

# In[108]:


df.boxplot()


# In[109]:


df.describe()


# In[110]:


q1=df["Profit"].describe()["25%"]


# In[111]:


q3=df["Profit"].describe()["75%"]


# In[112]:


iqr=q3-q1


# In[113]:


iqr


# # Observation  
# 1. Finded Outliers with by Visuvaling the graph Outlier Have in the SepalWidthCm farmat
# 
# 2. Finded Outlier by using mathemetics IQR Techniques of Entire data and found SepalWidthCm farmat,
# and divided to 25% & 75% Subtract the iq3-iq1 
# 
# ### Now let's find the Lower_ & Upper_boundary

# In[114]:


lower_boundary=q1-(0.8*iqr)
# lower boundary is taking 0.8 times distance


# In[115]:


upper_boundary=q3+(0.8*iqr)
# upper_boundary is taking 0.8 times distance


# #### After done with lower and upper boundary method let's find the outlier either have lower or upper side
# 

# In[116]:


#use .index function to find out the 
outlier_idx=df[(df["Profit"]<lower_boundary) | (df["Profit"]>upper_boundary)].index


# In[117]:


df.drop(outlier_idx)


# In[118]:


df.drop(outlier_idx,inplace=True)


# In[119]:


#check the outlier ether have or not 
df.boxplot()


# ## Encoding Categorical data 
# One Hot Encoding

# In[120]:


df=pd.get_dummies(df,columns=['State'])


# In[121]:


df.head()


# ## Split Dependent and Independent Variable

# In[122]:


x=df.iloc[:,[0,1,2,4,5,6]]
x 


# In[123]:


y=df.iloc[:,3]
y


# ## Split Train and test 

# In[124]:


from sklearn.model_selection import train_test_split


# In[125]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
# what is Random_state 
# what will hapend without Random_state, when will run multiple time every time it represent deferent values.
# once will use Random_state and we can run multiple time it can be represent same values


# In[126]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[127]:


sns.pairplot(df)


# ## Decision Tree no need a Scaling part 

# In[128]:


#from sklearn.preprocessing import MinMaxScaler


# In[129]:


#sc=MinMaxScaler()


# In[130]:


#x_train=sc.fit_transform(x_train)


# In[131]:


#x_test=sc.transform(x_test)


# # Build Model Decision Tree Regression 

# In[132]:


from sklearn.tree import DecisionTreeRegressor


# ### what are the Hyeparameters are in main here 

# In[198]:


# creaing the instance of model
# using parametr criterion="entropy" to calculate the Information gain
# another formal also have is Gini as a same like Entropy


dt_model=DecisionTreeRegressor(criterion='mse',max_depth=5) 
# dt_model=DecisionTreeRegressor(criterion='gini')


# In[199]:


# Train the model
dt_model.fit(x_train, y_train)


# ## Prediction

# #### Predict for Test 

# In[200]:


y_pred=dt_model.predict(x_test) # by defualt thrisold value given 0 to 1


# In[201]:


# orginal prabpliy value
y_pred=dt_model.predict(x_test)
# by left side of prab 0, by left side prab 1 


# In[202]:


y_pred


# In[203]:


y_test


# #### Predict 
# for Train but not Necessary to predict to Train 
# 
# y_pred_train=dt_model.predict(x_train)
# 

# # Evaluation

# In[204]:


from sklearn.metrics import r2_score


# In[205]:


r2_score(y_test,y_pred)


# In[206]:


from sklearn.metrics import mean_squared_error


# In[207]:


mse=mean_squared_error(y_test,y_pred)


# In[208]:


mse


# In[209]:


rmse=np.sqrt(mse)


# In[210]:


print('Test data Error ::',rmse)


# ### check the train also 

# In[211]:


y_pred_train=dt_model.predict(x_train) # input


# In[212]:


y_pred_train


# In[213]:


r2_score(y_train,y_pred_train)


# In[214]:


mse=mean_squared_error(y_train,y_pred_train)


# In[215]:


mse


# In[216]:


rmse=np.sqrt(mse)


# In[217]:


print('Train data Error ::',rmse)


# ## Plot the decision tree

# In[218]:


from sklearn.tree import plot_tree


# In[219]:


plot_tree(dt_model)
plt.show()


# In[220]:


x_train.columns


# In[221]:


plt.figure(figsize=(15,15))
plot_tree(dt_model,fontsize=9,feature_names=x_train.columns,filled=True)
plt.show()

# x_train.columns  for name on nodes

as per belove graph 
which mse value is high that will be Root node
mse is nothing but Avarage deviation

Split down till the mse will 0

MSE Avarage deviation formula 
MSE = 1/n  n∑i=1 (Yi - Y^i)2

MSE = mean squared error
n = number of data points
Yi = observed values
Yi = predicted values
# In[ ]:




