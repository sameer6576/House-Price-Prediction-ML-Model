
# coding: utf-8

# # Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("D:\ML Project\data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins = 50, figsize = (20,15))


# # Train-Test Splitting
# 

# In[8]:


import numpy as np

# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices] , data.iloc[test_indices]


# In[9]:


# train_set, test_set = split_train_test(housing , 0.2)


# In[10]:


# print(f"Rows in train set : {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set : {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[12]:


# stratified shuffling for equal distribution of data

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)
for train_index, test_index in split.split(housing , housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[13]:


# strat_train_set['CHAS'].value_counts()


# In[14]:


# strat_test_set['CHAS'].value_counts()


# In[15]:


# copy the training set right now before looking for correlations


# In[16]:


housing = strat_train_set.copy()


# # Looking for correlations
# 

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


# from pandas.plotting import scatter_matrix
# attributes  = ['MEDV' ,'RM' , 'ZN' , 'LSTAT']
# scatter_matrix(housing[attributes] , figsize = (12,8))


# In[19]:


# housing.plot(kind = "scatter" , x = "RM" , y = "MEDV" , alpha = 0.8)


# ## Trying out attribute combinations

# In[20]:


housing["TAXRM"] = housing['TAX'] / housing['RM']


# In[21]:


housing.head()


# In[22]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


# housing.plot(kind = "scatter" , x = "TAXRM" , y = "MEDV" , alpha = 0.8)


# In[24]:


housing = strat_train_set.drop("MEDV" , axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# ## missing attributes
# 

# In[25]:


# To take care of missing attributes we have 3 options:
# 1. Get rid of the missing data points (not feasible for large missing data)
# 2. Get rid of the whole attribute (only if the correlation is weak)
# 3. Set the value to some value(0,mean or median) (correct way here)


# In[26]:


#option 1
# note that no rm column is there and the original housing datafram will remain unchanged

a = housing.dropna(subset = ['RM'])
a.shape


# In[27]:


housing.drop("RM" , axis = 1).shape #option 2


# In[28]:


median = housing["RM"].median()


# In[29]:


housing["RM"].fillna(median)


# In[30]:


housing.shape


# In[31]:


housing.describe()


# In[32]:


from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(strategy = 'median')
imputer.fit(housing)


# In[33]:


imputer.statistics_


# In[34]:


X = imputer.transform(housing)


# In[35]:


housing_tr = pd.DataFrame(X,columns = housing.columns)


# In[36]:


housing_tr.describe()


# ## scikit-learn design

# Primarily there are three types of objects
# 1. estimators : it estimates some parameter based on a dataset eg. imputer
# it has a fit a method and transform method
# 2. transformers : transform method takes input and returns output based on the learnings from fit().
# 3. predicators : isme fit and predict method hota hain

# ## Feature scaling
# 

# two types of feature scaling methods hote hain
# 1. min max scaling (normalization)
#     (value-min)/(max-min)
#     sklearn ki class hain ek called MinMaxScaler for this
# 2. standardization
#     (value-mean)/std
#     sklearn provides a class called standard scaler for this
#     

# 
# ## Creating a Pipeline

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy = "median")) , 
    ('std_scalar' , StandardScaler()),
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[39]:


housing_num_tr.shape


# ## Selecting a desired model for dragon real estates

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[41]:


some_data = housing.iloc[:5]


# In[42]:


some_labels = housing_labels.iloc[:5]


# In[43]:


prepared_data = my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[58]:


prepared_data[0]


# In[45]:


list(some_labels)


# ## Evaluating the model

# In[46]:


from sklearn.metrics import mean_squared_error
housing_predictions=  model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[47]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[48]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr, housing_labels, scoring = "neg_mean_squared_error",cv = 10)
rmse_scores = np.sqrt(-scores)


# In[49]:


rmse_scores


# In[50]:


def print_scores(scores):
    print("Scores:" , scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


# In[51]:


print_scores(rmse_scores)


# ##  Quiz : convert this notebook into python file and run the pipeline using visual studio code

# ## saving the model

# In[52]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# ## Testing the model on test data

# In[56]:


X_test = strat_test_set.drop("MEDV" , axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[54]:


final_rmse


# In[ ]:




