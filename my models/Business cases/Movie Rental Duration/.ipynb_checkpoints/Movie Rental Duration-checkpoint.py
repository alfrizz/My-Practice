#!/usr/bin/env python
# coding: utf-8

# ![dvd_image](dvd_image.jpg)
# 
# A DVD rental company needs your help! They want to figure out how many days a customer will rent a DVD for based on some features and has approached you for help. They want you to try out some regression models which will help predict the number of days a customer will rent a DVD for. The company wants a model which yeilds a MSE of 3 or less on a test set. The model you make will help the company become more efficient inventory planning.
# 
# The data they provided is in the csv file `rental_info.csv`. It has the following features:
# - `"rental_date"`: The date (and time) the customer rents the DVD.
# - `"return_date"`: The date (and time) the customer returns the DVD.
# - `"amount"`: The amount paid by the customer for renting the DVD.
# - `"amount_2"`: The square of `"amount"`.
# - `"rental_rate"`: The rate at which the DVD is rented for.
# - `"rental_rate_2"`: The square of `"rental_rate"`.
# - `"release_year"`: The year the movie being rented was released.
# - `"length"`: Lenght of the movie being rented, in minuites.
# - `"length_2"`: The square of `"length"`.
# - `"replacement_cost"`: The amount it will cost the company to replace the DVD.
# - `"special_features"`: Any special features, for example trailers/deleted scenes that the DVD also has.
# - `"NC-17"`, `"PG"`, `"PG-13"`, `"R"`: These columns are dummy variables of the rating of the movie. It takes the value 1 if the move is rated as the column name and 0 otherwise. For your convinience, the reference dummy has already been dropped.

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# Import any additional modules and start coding below


# In[2]:


rental = pd.read_csv('rental_info.csv')
rental


# In[3]:


rental.info()


# In[4]:


rental.special_features.unique()


# In[5]:


rental['Deleted Scenes'] = [1 if 'Deleted Scenes' in rental.special_features[i] else 0 for i in range(len(rental))]
rental['Behind the Scenes'] = [ 1 if 'Behind the Scenes' in rental.special_features[i] else 0 for i in range(len(rental))]
rental.drop('special_features', axis = 1, inplace = True)
rental


# In[6]:


rental['rental_date'] = pd.to_datetime(rental['rental_date'])
rental['return_date'] = pd.to_datetime(rental['return_date'])
rental.info()


# In[7]:


rental['y'] = (rental['return_date'] - rental['rental_date']).dt.days
rental.drop(['return_date','rental_date'], axis = 1, inplace = True)
rental


# In[8]:


sns.pairplot(rental[['amount', 'length', 'rental_rate', 'y']])


# In[ ]:


sns.pairplot(rental[['amount_2', 'length_2', 'rental_rate_2', 'y']])


# In[ ]:


rental.shape


# In[ ]:


X = rental.drop('y', axis = 1)
X


# In[ ]:


y = rental['y']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform both the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


# Initialize the Ridge regression model with a penalty
ridge = Ridge(alpha=1.0)

# Fit the model
ridge.fit(X_train_scaled, y_train)

# Make predictions
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_ridge


# In[ ]:


min(y_test), min(ridge_predictions), max(y_test), max(ridge_predictions)


# In[ ]:


sns.histplot(y_test)


# In[ ]:


sns.histplot(y_pred_ridge)


# In[ ]:


sns.regplot(x=y_test, y=y_pred_ridge, scatter_kws={'s':10}, line_kws={'color':'red'})
# Add the 45-degree reference line
max_val = max(max(y_test), max(y_pred_ridge))
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2)


# In[ ]:


mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_ridge


# In[ ]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=777)
rf_regressor.fit(X_train_scaled, y_train)
y_pred_rf = rf_regressor.predict(X_test_scaled)
y_pred_rf


# In[ ]:


sns.histplot(y_pred_rf)


# In[ ]:


sns.regplot(x=y_test, y=y_pred_rf, scatter_kws={'s':10}, line_kws={'color':'red'})
# Add the 45-degree reference line
max_val = max(max(y_test), max(y_pred_rf))
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2)


# In[ ]:


mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_rf


# In[ ]:


xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=777)

# Train the model
xg_reg.fit(X_train_scaled, y_train)

y_pred_xgb = xg_reg.predict(X_test_scaled)
y_pred_xgb


# In[ ]:


sns.histplot(y_pred_xgb)


# In[ ]:


sns.regplot(x=y_test, y=y_pred_xgb, scatter_kws={'s':10}, line_kws={'color':'red'})
# Add the 45-degree reference line
max_val = max(max(y_test), max(y_pred_rf))
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2)


# In[ ]:


mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mse_xgb


# In[ ]:


# XGBoost gives lowest MSE so:
best_model = xg_reg
best_mse = mse_xgb

