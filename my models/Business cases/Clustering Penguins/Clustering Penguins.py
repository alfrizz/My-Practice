#!/usr/bin/env python
# coding: utf-8

# ![Alt text](https://imgur.com/orZWHly.png=80)
# source: @allison_horst https://github.com/allisonhorst/penguins

# You have been asked to support a team of researchers who have been collecting data about penguins in Antartica! The data is available in csv-Format as `penguins.csv`
# 
# **Origin of this data** : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
# 
# **The dataset consists of 5 columns.**
# 
# Column | Description
# --- | ---
# culmen_length_mm | culmen length (mm)
# culmen_depth_mm | culmen depth (mm)
# flipper_length_mm | flipper length (mm)
# body_mass_g | body mass (g)
# sex | penguin sex
# 
# Unfortunately, they have not been able to record the species of penguin, but they know that there are **at least three** species that are native to the region: **Adelie**, **Chinstrap**, and **Gentoo**.  Your task is to apply your data science skills to help them identify groups in the dataset!

# In[1]:


# Import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from kmodes.kprototypes import KPrototypes
from scipy import stats
from scipy.stats import shapiro, boxcox

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[2]:


# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()


# In[3]:


penguins_df.info()


# In[4]:


penguins_df.groupby('sex').agg({
    'culmen_length_mm'   : ['mean', 'std'],
    'culmen_depth_mm'    : ['mean', 'std'],
    'flipper_length_mm'  : ['mean', 'std'],
    'body_mass_g'        : ['mean', 'std'],
})


# In[5]:


num_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']


# In[6]:


for feature in num_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=penguins_df, x='sex', y=feature)


# In[7]:


for feature in num_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=penguins_df, x=feature)


# In[8]:


# normality check

# Apply Shapiro-Wilk test to each numeric feature
for feature in num_features:
    stat, p = shapiro(penguins_df[feature])
    print(f'Shapiro-Wilk Test for {feature}: Statistics={stat}, p-value={p}')
    if p <= 0.05:
        print('data NOT normally distributed')
    else:
        print('data YES normally distributed')


# In[9]:


# Q-Q plot
for feature in num_features:
    stats.probplot(penguins_df[feature], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {feature}')
    plt.show()


# In[10]:


# # Apply Box-Cox transformation to each numeric feature, 

# for feature in num_features:
#     # Ensure all values are positive by adding a small constant if necessary
#     penguins_df[feature] = penguins_df[feature] + 1e-6
#     penguins_df[feature], _ = boxcox(penguins_df[feature])


# In[11]:


# for feature in num_features:
#     plt.figure(figsize=(8, 6))
#     sns.histplot(data=penguins_df, x=feature)


# In[12]:


# # normality check

# # Apply Shapiro-Wilk test to each numeric feature
# for feature in num_features:
#     stat, p = shapiro(penguins_df[feature])
#     print(f'Shapiro-Wilk Test for {feature}: Statistics={stat}, p-value={p}')
#     if p <= 0.05:
#         print('data NOT normally distributed')
#     else:
#         print('data YES normally distributed')


# In[13]:


# # Q-Q plot
# for feature in num_features:
#     stats.probplot(penguins_df[feature], dist="norm", plot=plt)
#     plt.title(f'Q-Q Plot for {feature}')
#     plt.show()


# In[14]:


penguins_df.info()


# In[15]:


# as the data is not too far from a normal distribution, standardization might be more appropriate as it centers the data and adjusts for variance.

scaler = StandardScaler()
stand_num_features = scaler.fit_transform(penguins_df[num_features])
penguins_df[num_features] = stand_num_features

penguins_df


# In[16]:


# Convert categorical feature to numerical codes
penguins_df['sex'] = penguins_df['sex'].astype('category').cat.codes

penguins_df


# In[17]:


# Convert DataFrame to numpy array
penguins_array = penguins_df.to_numpy()
penguins_array


# In[18]:


# ELBOW Method

cost = [] # “cost” in K-Prototypes is analogous to “inertia” in K-Means
silhouette = []

for k in range(1, 11):
    kproto = KPrototypes(n_clusters=k, init='Cao', verbose=0)
    kproto.fit(penguins_array, categorical=[4]) # categorical=[4] indicates that the 5th column (index 4) is a categorical feature (in this case, the sex attribute).
    cost.append(kproto.cost_)
    print(f"The inertia (cost) with {k} clusters is: {kproto.cost_}")
    if k > 1:
        cluster_labels = kproto.predict(penguins_array, categorical=[4]) 
        # Calculate the Silhouette Score
        silhouette_avg = silhouette_score(penguins_array, cluster_labels)
        print(f"The average silhouette score with {k} clusters is: {silhouette_avg}\n")
        silhouette.append(silhouette_avg)


# In[19]:


# Plot the cost values
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(1, 11), cost, marker='o')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Cost')
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette score')
plt.plot(range(2, 11), silhouette, marker='x', c='orange')
plt.title('Elbow Method for Optimal k with K-Prototypes')
plt.show()


# In[20]:


# Apply K-Prototypes
kproto = KPrototypes(n_clusters=5, init='Cao', verbose=1)
cluster_labels_KM = kproto.fit_predict(penguins_array, categorical=[4]) 
# Add cluster labels to the DataFrame
penguins_df['cluster_KM'] = cluster_labels_KM
penguins_df


# In[21]:


sns.scatterplot(data=penguins_df, x='culmen_length_mm', y='culmen_depth_mm',hue='cluster_KM',palette="viridis")


# In[22]:


sns.scatterplot(data=penguins_df, x='flipper_length_mm', y='body_mass_g',hue='cluster_KM',palette="viridis")


# In[23]:


sns.scatterplot(data=penguins_df, x='sex', y='body_mass_g',hue='cluster_KM',palette="viridis")


# In[24]:


sns.scatterplot(data=penguins_df, x='culmen_length_mm', y='culmen_depth_mm',hue='sex')


# In[25]:


# DBSCAN

minPoints = 7 # The value of minPoints is often set to twice the number of dimensions of the dataset (2*penguins_array.shape[1])

# Compute the k-distance (k = minPoints - 1)
neighbors = NearestNeighbors(n_neighbors=minPoints)
neighbors_fit = neighbors.fit(penguins_array)
distances, indices = neighbors_fit.kneighbors(penguins_array)

# Set print options to display all columns but limit rows
np.set_printoptions(threshold=0, edgeitems=4, linewidth=np.inf)

print(distances.shape, indices.shape)
print(distances)
print(indices)


# In[26]:


# Sort the distances of all the kth (minPonnts - 1) neighbours
distances = np.sort(distances[:, minPoints - 1])

# Plot the k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('k-Distance Graph')
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{minPoints}-th Nearest Neighbor Distance')
plt.show()


# In[27]:


epsilon = 0.8  # set at the point where the above distance curve forms a elbow

# Apply DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=minPoints)
cluster_labels_DBS = dbscan.fit_predict(penguins_array)
penguins_df['cluster_DBS'] = cluster_labels_DBS + 1 # otherwise the clusters starts from -1
print('number of DBSCAN clusters =', penguins_df['cluster_DBS'].nunique())
penguins_df


# In[28]:


sns.scatterplot(data=penguins_df, x='culmen_length_mm', y='culmen_depth_mm',hue='cluster_DBS',palette="viridis")


# In[29]:


all_features = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']

# Calculate the correlation matrix
penguins_df[all_features].corr()


# In[30]:


# continuous_features = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g']
# df_continuous_features = penguins_df[continuous_features]

df_all_features = penguins_df[all_features]

X = add_constant(df_all_features)  # Add a constant term for the intercept to check multicollinearity with Variance Inflation Factor (VIF)
X


# In[31]:


vif = pd.DataFrame()
vif['Variable'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# In[32]:


# checking linearity

sns.pairplot(df_all_features)


# In[33]:


# given the features linearity, we apply PCA for dimensionality reduction

max_components = len(all_features)
total_explained_variance = []

for n_components in range(1, max_components + 1):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(penguins_df[all_features])
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance.append(np.sum(explained_variance_ratio))
    
total_explained_variance


# In[34]:


plt.plot(range(1, max_components + 1), total_explained_variance, marker='o')
plt.title('Total Variance Percentage explained by the Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Total Variance Percentage explained')


# In[35]:


# let's use 3 PCs, which explain more than 95% of the total variance percentage

pca = PCA(n_components = 3)
principal_components = pca.fit_transform(penguins_df[all_features])
df_penguins_PCA = pd.DataFrame(principal_components, columns = ['PC1', 'PC2', 'PC3'])
df_penguins_PCA['cluster_KM'] = cluster_labels_KM
df_penguins_PCA['cluster_DBS'] = cluster_labels_DBS + 1
df_penguins_PCA


# In[36]:


sns.scatterplot(data=df_penguins_PCA, x='PC1', y='PC2',hue='cluster_KM',palette="viridis")


# In[37]:


sns.scatterplot(data=df_penguins_PCA, x='PC1', y='PC2',hue='cluster_DBS',palette="viridis")


# In[38]:


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(df_penguins_PCA['PC1'], df_penguins_PCA['PC2'], df_penguins_PCA['PC3'], 
                     c=df_penguins_PCA['cluster_KM'], cmap='viridis')

# Add labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Add a legend
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)


# In[39]:


# The output should be a DataFrame named stat_penguins with one row per cluster that shows the mean of the original variables (or columns in "penguins.csv") by cluster. stat_penguins should not include any non-numeric columns.


# In[44]:


stat_penguins = penguins_df.groupby('cluster_KM')[['culmen_length_mm','culmen_depth_mm','flipper_length_mm']].mean().rename_axis('label')
stat_penguins


# In[ ]:





# In[ ]:





# In[ ]:




