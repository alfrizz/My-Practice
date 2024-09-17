#!/usr/bin/env python
# coding: utf-8

# ![car](car.jpg)
# 
# Insurance companies invest a lot of [time and money](https://www.accenture.com/_acnmedia/pdf-84/accenture-machine-leaning-insurance.pdf) into optimizing their pricing and accurately estimating the likelihood that customers will make a claim. In many countries insurance it is a legal requirement to have car insurance in order to drive a vehicle on public roads, so the market is very large!
# 
# Knowing all of this, On the Road car insurance have requested your services in building a model to predict whether a customer will make a claim on their insurance during the policy period. As they have very little expertise and infrastructure for deploying and monitoring machine learning models, they've asked you to identify the single feature that results in the best performing model, as measured by accuracy, so they can start with a simple model in production.
# 
# They have supplied you with their customer data as a csv file called `car_insurance.csv`, along with a table detailing the column names and descriptions below.

# 
# 
# ## The dataset
# 
# | Column | Description |
# |--------|-------------|
# | `id` | Unique client identifier |
# | `age` | Client's age: <br> <ul><li>`0`: 16-25</li><li>`1`: 26-39</li><li>`2`: 40-64</li><li>`3`: 65+</li></ul> |
# | `gender` | Client's gender: <br> <ul><li>`0`: Female</li><li>`1`: Male</li></ul> |
# | `driving_experience` | Years the client has been driving: <br> <ul><li>`0`: 0-9</li><li>`1`: 10-19</li><li>`2`: 20-29</li><li>`3`: 30+</li></ul> |
# | `education` | Client's level of education: <br> <ul><li>`0`: No education</li><li>`1`: High school</li><li>`2`: University</li></ul> |
# | `income` | Client's income level: <br> <ul><li>`0`: Poverty</li><li>`1`: Working class</li><li>`2`: Middle class</li><li>`3`: Upper class</li></ul> |
# | `credit_score` | Client's credit score (between zero and one) |
# | `vehicle_ownership` | Client's vehicle ownership status: <br><ul><li>`0`: Does not own their vehilce (paying off finance)</li><li>`1`: Owns their vehicle</li></ul> |
# | `vehcile_year` | Year of vehicle registration: <br><ul><li>`0`: Before 2015</li><li>`1`: 2015 or later</li></ul> |
# | `married` | Client's marital status: <br><ul><li>`0`: Not married</li><li>`1`: Married</li></ul> |
# | `children` | Client's number of children |
# | `postal_code` | Client's postal code | 
# | `annual_mileage` | Number of miles driven by the client each year |
# | `vehicle_type` | Type of car: <br> <ul><li>`0`: Sedan</li><li>`1`: Sports car</li></ul> |
# | `speeding_violations` | Total number of speeding violations received by the client | 
# | `duis` | Number of times the client has been caught driving under the influence of alcohol |
# | `past_accidents` | Total number of previous accidents the client has been involved in |
# | `outcome` | Whether the client made a claim on their car insurance (response variable): <br><ul><li>`0`: No claim</li><li>`1`: Made a claim</li></ul> |

# In[1]:


# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Start coding!


# Identify the single feature of the data that is the best predictor of whether a customer will put in a claim (the "outcome" column), excluding the "id" column.
# 
# Store as a DataFrame called best_feature_df, containing columns named "best_feature" and "best_accuracy" with the name of the feature with the highest accuracy, and the respective accuracy score.

# In[2]:


df = pd.read_csv('car_insurance.csv')
df


# In[3]:


df.info()


# In[4]:


for col in df.columns:
    print(col, df[col].value_counts(dropna=False))
    print('-------')


# In[5]:


binary_features = ['vehicle_year','vehicle_type']


# In[6]:


categ_non_ord_features = ['postal_code']


# In[7]:


categ_ord_features = ['driving_experience','education','income']


# In[8]:


for feature in binary_features:
    df[feature] = df[feature].astype('category').cat.rename_categories([0,1]).astype('int64')


# In[9]:


driving_experience_map = {'0-9y':0, '10-19y':1, '20-29y':2, '30y+':3}
education_map = {'high school':1, 'none':0, 'university':2}
income_map = {'upper class':3, 'poverty':0, 'working class':1, 'middle class':2}


# In[10]:


for feature in categ_ord_features:
    df[feature] = df[feature].map(globals()[str(feature)+'_map'])


# In[11]:


# frequency encoding

postal_code_frequency = df.postal_code.value_counts(normalize = True)
print(postal_code_frequency)

for feature in categ_non_ord_features:
    df[feature] = df[feature].map(postal_code_frequency)

df.postal_code.value_counts()


# In[12]:


# # target encoding 

# postal_code_target_mean = df.groupby('postal_code')['outcome'].mean()
# print(postal_code_target_mean)

# for feature in categ_non_ord_features:
#     df[feature] = df[feature].map(postal_code_target_mean)

# df.postal_code.value_counts()


# In[13]:


# # one hot encoding

# df = pd.get_dummies(df, columns=['postal_code'])
# df


# In[14]:


df['credit_score'].fillna(df['credit_score'].mean(), inplace=True)
df['annual_mileage'].fillna(df['annual_mileage'].mode()[0], inplace=True)


# In[15]:


df.info()


# In[16]:


# y = df['outcome']
# X = df.drop(['outcome','id'], axis=1)


# In[17]:


train_data, test_data = train_test_split(df, test_size=0.25, random_state=777)


# In[18]:


features = df.drop(['outcome','id'], axis=1).columns
features


# In[19]:


models = [] # saving a fitted model for each feature

for feature in features:
    model = logit(f"outcome ~ {feature}", data=train_data).fit()
    models.append(model)


# In[20]:


coefficients = {}
accuracies = {}

for i, feature in enumerate(features):
    # print(models[i].summary())
    coefficients[feature] = models[i].params[1]
    predictions = models[i].predict(test_data)
    conf_matrix = models[i].pred_table()
    # print(conf_matrix)
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    acc = (tn + tp) / (tn + fn + fp + tp)
    accuracies[feature] = acc


# In[21]:


sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
sorted_accuracies


# In[22]:


# # Create best_feature_df
# best_feature_df = pd.DataFrame({"best_feature": best_feature,
#                                 "best_accuracy": max(accuracies)},
#                                 index=[0])

best_feature = next(iter(sorted_accuracies.items()))[0]
best_accuracy = next(iter(sorted_accuracies.items()))[1]

best_feature_df = pd.DataFrame({'best_feature': best_feature, 
                               'best_accuracy': best_accuracy},
                              index=[0])
best_feature_df


# In[23]:


sorted_coefficients_abs = dict(sorted(coefficients.items(), key=lambda item: np.abs(item[1]), reverse=True))
sorted_coefficients_abs


# In[24]:


all_features_formula = "outcome ~ " + " + ".join(features)
all_features_formula


# In[25]:


all_features_model = logit(all_features_formula, data=train_data).fit()
all_features_model.summary()


# In[26]:


all_features_predictions = all_features_model.predict(test_data)

# Convert predictions to binary outcomes
predicted_classes = (all_features_predictions > 0.5).astype(int)

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_data["outcome"], predicted_classes)
tn, fp, fn, tp = conf_matrix.ravel()

# Compute accuracy
accuracy = (tn + tp) / (tn + fn + fp + tp)
print(f"Accuracy: {accuracy}")

# Print coefficients
coefficients = all_features_model.params
print("Coefficients:")
print(coefficients)

