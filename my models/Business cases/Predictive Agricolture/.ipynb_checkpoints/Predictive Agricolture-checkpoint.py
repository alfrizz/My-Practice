#!/usr/bin/env python
# coding: utf-8

# # Sowing Success: How Machine Learning Helps Farmers Select the Best Crops
# 
# ![Farmer in a field](farmer_in_a_field.jpg)
# 
# Measuring essential soil metrics such as nitrogen, phosphorous, potassium levels, and pH value is an important aspect of assessing soil condition. However, it can be an expensive and time-consuming process, which can cause farmers to prioritize which metrics to measure based on their budget constraints.
# 
# Farmers have various options when it comes to deciding which crop to plant each season. Their primary objective is to maximize the yield of their crops, taking into account different factors. One crucial factor that affects crop growth is the condition of the soil in the field, which can be assessed by measuring basic elements such as nitrogen and potassium levels. Each crop has an ideal soil condition that ensures optimal growth and maximum yield.
# 
# A farmer reached out to you as a machine learning expert for assistance in selecting the best crop for his field. They've provided you with a dataset called `soil_measures.csv`, which contains:
# 
# - `"N"`: Nitrogen content ratio in the soil
# - `"P"`: Phosphorous content ratio in the soil
# - `"K"`: Potassium content ratio in the soil
# - `"pH"` value of the soil
# - `"crop"`: categorical values that contain various crops (target variable).
# 
# Each row in this dataset represents various measures of the soil in a particular field. Based on these measurements, the crop specified in the `"crop"` column is the optimal choice for that field.  
# 
# In this project, you will build multi-class classification models to predict the type of `"crop"` and identify the single most importance feature for predictive performance.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, f_oneway


# In[2]:


# Load the dataset
crops = pd.read_csv("soil_measures.csv")
crops


# In[3]:


crops.info()


# In[4]:


crops['crop'].value_counts()


# In[5]:


correlation_matrix = crops.corr()
correlation_matrix


# In[6]:


# visualize

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[7]:


# applying anova to check correlation of target categories

#Perform Levene's test to check homoscedasticity first
categories = crops['crop'].unique()

features = crops.columns[:-1]
features


# In[8]:


for feat in features:
    samples = [crops[crops['crop'] == cat][feat] for cat in categories]
    stat, p_value = levene(*samples)
    print(f'Levene\'s Test for feature {feat}: Statistic={stat}, p-value={p_value}')
    
# a low p-value indicates that the variances of the 'ph' feature groups (for each category of the target 'crop') are significantly different, so we can't apply Anova for the feature 'ph'


# In[9]:


features_anova = features.drop('ph')
features_anova


# In[10]:


anova_results = {feat: f_oneway(*(crops[crops['crop'] == cat][feat] for cat in categories)) for feat in features_anova}

for feature, result in anova_results.items():
    print(f'Feature: {feature}, F-statistic: {result.statistic}, p-value: {result.pvalue}')

# Since all the p-values are 0.0, it means that the differences in the means of the features ‘N’, ‘P’, and ‘K’ across the different categories of ‘crop’ are statistically significant. This suggests that these features are important in distinguishing between the different categories of the target variable.


# In[11]:


# plot features distribution

for feature in features:
    sns.histplot(data=crops, x=feature, kde=True)
    plt.title('Distribution of Feature'+ feature)
    plt.show()


# In[12]:


# using standardization to scale the features, and make them more normally distributed

X = crops.drop(columns=['crop'])
y = crops['crop']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

# crops_scaled = pd.DataFrame(X_scaled, columns = features)
# crops_scaled


# In[13]:


# for feature in features:
#     sns.histplot(data=crops_scaled, x=feature, kde=True)
#     plt.title('Distribution of Features after Scaling'+ feature)
#     plt.show()


# In[14]:


# Set up the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, penalty='l2', C= 1)

# Set up cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {scores}')
print(f'Mean Accuracy: {np.mean(scores)}')
print(f'Standard Deviation: {np.std(scores)}')


# In[15]:


# Identify the single feature that has the strongest predictive performance for classifying crop types.

# Find the feature in the dataset that produces the best score for predicting "crop".
# From this information, create a variable called best_predictive_feature, which:
# Should be a dictionary containing the best predictive feature name as a key and the evaluation score (for the metric you chose) as the value.


# In[16]:


predictions_per_feature = {}

for i in range(len(X_scaled[0])):
    # Perform cross-validation one features at a time
    scores = cross_val_score(model, X_scaled[:,i].reshape(-1, 1), y, cv=skf, scoring='f1_macro')
    print('Results for feature:' + features[i])
    print(f'Cross-Validation F1 Scores: {scores}')
    mean_score = np.mean(scores)
    print(f'Mean F1: {mean_score}')
    print(f'Standard Deviation: {np.std(scores)}')
    predictions_per_feature[features[i]] = mean_score
print('\n',predictions_per_feature)


# In[17]:


# Find the key with the maximum value
best_predictive_key = max(predictions_per_feature, key=predictions_per_feature.get)

# Extract the key-value pair
best_predictive_feature = {best_predictive_key, predictions_per_feature[best_predictive_key]}

best_predictive_key, best_predictive_feature


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




