#!/usr/bin/env python
# coding: utf-8

# ![A soccer pitch for an international match.](soccer-pitch.jpg)
# 
# You're working as a sports journalist at a major online sports media company, specializing in soccer analysis and reporting. You've been watching both men's and women's international soccer matches for a number of years, and your gut instinct tells you that more goals are scored in women's international football matches than men's. This would make an interesting investigative article that your subscribers are bound to love, but you'll need to perform a valid statistical hypothesis test to be sure!
# 
# While scoping this project, you acknowledge that the sport has changed a lot over the years, and performances likely vary a lot depending on the tournament, so you decide to limit the data used in the analysis to only official `FIFA World Cup` matches (not including qualifiers) since `2002-01-01`.
# 
# You create two datasets containing the results of every official men's and women's international football match since the 19th century, which you scraped from a reliable online source. This data is stored in two CSV files: `women_results.csv` and `men_results.csv`.
# 
# The question you are trying to determine the answer to is:
# 
# > Are more goals scored in women's international soccer matches than men's?
# 
# You assume a **10% significance level**, and use the following null and alternative hypotheses:
# 
# $H_0$ : The mean number of goals scored in women's international soccer matches is the same as men's.
# 
# $H_A$ : The mean number of goals scored in women's international soccer matches is greater than men's.

# In[73]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, levene, t, mannwhitneyu


# In[2]:


men_results = pd.read_csv('men_results.csv', index_col=0)
men_results['tot_score'] = men_results['home_score'] + men_results['away_score'] 
men_results['gender'] = 'Men'
men_results


# In[3]:


women_results = pd.read_csv('women_results.csv', index_col=0)
women_results['tot_score'] = women_results['home_score'] + women_results['away_score'] 
women_results['gender'] = 'Women'
women_results


# In[4]:


# common tournaments between men and women

men_tournaments = men_results['tournament'].unique()
women_tournaments = women_results['tournament'].unique()

common_tournaments = [tournament for tournament in men_tournaments if tournament in women_tournaments]
common_tournaments


# In[5]:


# create a dataframe joining men and women

results = pd.concat([men_results, women_results])
results['common_tournament'] = results['tournament'].isin(common_tournaments)
results


# In[53]:


results['date'] = pd.to_datetime(results['date'])
results.info()


# In[6]:


# average total score by common tournaments

grouped_scores = results[results['common_tournament']==True].groupby(['gender', 'tournament'])['tot_score'].mean()

grouped_scores = grouped_scores.reset_index() # Reset the index to convert the grouped data into a DataFrame
grouped_scores


# In[7]:


# Unstack the DataFrame (long to wide), to add the score_difference column

# Set the index to ['tournament', 'gender']
grouped_scores.set_index(['tournament', 'gender'], inplace=True)

# Unstack the 'gender' level
unstacked_grouped_scores = grouped_scores.unstack(level='gender')

# Rename the columns
unstacked_grouped_scores.columns = ['tot_score_men', 'tot_score_women']

# Reset the index to make 'tournament' a column again
unstacked_grouped_scores.reset_index(inplace=True)

# Calculate tot_score_difference between woment and men
unstacked_grouped_scores['tot_score_diff'] = unstacked_grouped_scores['tot_score_women'] - unstacked_grouped_scores['tot_score_men']
unstacked_grouped_scores


# In[8]:


# Melt again the DataFrame (wide to long) for plotting

melted_grouped_scores = unstacked_grouped_scores.melt(id_vars=['tournament','tot_score_diff'], value_vars=['tot_score_men', 'tot_score_women'], var_name='gender', value_name='tot_score')
melted_grouped_scores

# Rename the 'gender' values for better readability
melted_grouped_scores['gender'] = melted_grouped_scores['gender'].replace({'tot_score_men': 'Men', 'tot_score_women': 'Women'})

# sort the dataframe by tot_score_diff for better visualization
melted_grouped_scores = melted_grouped_scores.sort_values(by='tot_score_diff', ascending=False)
melted_grouped_scores


# In[9]:


# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x='tournament', y='tot_score', hue='gender', hue_order=['Women', 'Men'], data=melted_grouped_scores)

plt.xticks(rotation=90)
plt.xlabel('Tournament')
plt.ylabel('Average Score')
plt.title('Average Score by Tournament for Men and Women')
plt.legend(title='Gender')
plt.show()


# In[10]:


gender_counts = results['gender'].value_counts()

plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')


# In[11]:


top_categories = results['tournament'].value_counts().nlargest(10).index
top_categories_men = men_results['tournament'].value_counts().nlargest(10).index
top_categories_women = women_results['tournament'].value_counts().nlargest(10).index

top_categories


# In[12]:


sns.countplot(x = results[results['tournament'].isin(top_categories)]['tournament'], order=top_categories)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 tournament categories All Genders')


# In[13]:


sns.countplot(x = men_results[men_results['tournament'].isin(top_categories_men)]['tournament'], order=top_categories_men)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 tournament categories Men')


# In[14]:


sns.countplot(x = women_results[women_results['tournament'].isin(top_categories_women)]['tournament'], order=top_categories_women)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 tournament categories Women')


# In[15]:


# There are no many friendly competitions for women. So we create another dataframe 'men_results_competitions', eliminating the category 'Friendly' from the men tournaments

men_results_compet = men_results[men_results['tournament'] != 'Friendly']

men_results_compet


# In[16]:


top_categories_men_compet = men_results_compet['tournament'].value_counts().nlargest(10).index
sns.countplot(x = men_results_compet[men_results_compet['tournament'].isin(top_categories_men_compet)]['tournament'], order=top_categories_men_compet)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 tournament categories Men - Only Competitions')


# In[17]:


# create a dataframe joining men (only competitions) and women

results_compet = pd.concat([men_results_compet, women_results])
results_compet


# In[18]:


gender_counts_compet = results_compet['gender'].value_counts()

plt.pie(gender_counts_compet, labels=gender_counts_compet.index, autopct='%1.1f%%')


# In[29]:


# apply chi square goodness of fit to check if there is a significant imbalance between men and women

# Count the number of records for each gender
gender_counts = results['gender'].value_counts()
print('actual counts:', gender_counts, '\n****************************************')

# Define the expected distribution (e.g., equal distribution)
expected_counts = [np.ceil(len(results) / 2), np.floor(len(results) / 2)]
print('expected counts:', expected_counts)


# In[33]:


# Perform the Chi-Square Goodness of Fit Test
chi2, p = chisquare(f_obs=gender_counts, f_exp=expected_counts)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")

# Interpret the result
if p < 0.05:
    print("There is a statistically significant imbalance between the number of records for men and women.")
else:
    print("There is no statistically significant imbalance between the number of records for men and women.")


# In[39]:


# apply Levene test to check for heteroscedasticity, then decide if to use the Welch or Student T test

# Separate the scores by gender
men_scores = results[results['gender'] == 'Men']['tot_score']
women_scores = results[results['gender'] == 'Women']['tot_score']

# Perform Levene's test
stat, p_value = levene(men_scores, women_scores)

print(f"Levene's Test Statistic: {stat}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("The variances are significantly different.")
else:
    print("The variances are not significantly different.")


# In[43]:


men_scores_size = len(men_scores)
women_scores_size = len(women_scores)

print('men sample size:', men_scores_size)
print('women sample mean:', women_scores_size)

men_scores_mean = men_scores.mean()
women_scores_mean = women_scores.mean()

print('men mean:', men_scores_mean)
print('women mean:', women_scores_mean)

men_scores_var = men_scores.var()
women_scores_var = women_scores.var()

print('men variance:', men_scores_var)
print('women variance:', women_scores_var)


# In[47]:


# given a statistically significant difference in sample size and in variance, we apply a weighted Welch's independent t-test

# Calculate the t-statistic
t_stat = (men_scores_mean - women_scores_mean) / np.sqrt((men_scores_var / men_scores_size) + (women_scores_var / women_scores_size))

# Calculate the degrees of freedom using the Welch-Satterthwaite equation
DoF = ((men_scores_var / men_scores_size) + (women_scores_var / women_scores_size))**2 / (((men_scores_var / men_scores_size)**2 / (men_scores_size - 1)) + ((women_scores_var / women_scores_size)**2 / (women_scores_size - 1)))

# Calculate the p-value (one tailed test, to check if the test statistic is grater than the critical value)
p_value = 1 - t.cdf(t_stat, DoF)

print(f"T-Statistic: {t_stat}")
print(f"Degrees of Freedom: {DoF}")
print(f"P-Value: {p_value}")


# In[50]:


# Interpret the result
if p_value < 0.10:
    print("There is a statistically significant difference in the average number of goals between men and women.")
else:
    print("There is no statistically significant difference in the average number of goals between men and women.")


# In[58]:


results


# In[62]:


# filtering the data for only "FIFA World Cup" and from 2002

results_filtered = results[(results['date'] >= '2002') & (results['tournament'] == 'FIFA World Cup')]
results_filtered


# In[64]:


men_scores_filt = results_filtered[results_filtered['gender'] == 'Men']['tot_score']
women_scores_filt = results_filtered[results_filtered['gender'] == 'Women']['tot_score']

filt_men_scores_size = len(men_scores_filt)
filt_women_scores_size = len(women_scores_filt)

print('men sample size:', filt_men_scores_size)
print('women sample mean:', filt_women_scores_size)

filt_men_scores_mean = men_scores_filt.mean()
filt_women_scores_mean = women_scores_filt.mean()

print('men mean:', filt_men_scores_mean)
print('women mean:', filt_women_scores_mean)

filt_men_scores_var = men_scores_filt.var()
filt_women_scores_var = women_scores_filt.var()

print('men variance:', filt_men_scores_var)
print('women variance:', filt_women_scores_var)


# In[65]:


gender_counts_filt = results_filtered['gender'].value_counts()

plt.pie(gender_counts_filt, labels=gender_counts_filt.index, autopct='%1.1f%%')


# In[72]:


# check distribution of the outcome, to determine it to apply parametric or non-parametric statistical tests

sns.histplot(men_scores_filt, label='Men', kde=True)
sns.histplot(women_scores_filt, label='Women', kde=True)
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.title('Histogram of Scores by Gender')
plt.legend()


# In[82]:


# Perform the Mann-Whitney U test (right-tailed: checking if women_scores_filt is greater than men_scores_filt)
stat, p = mannwhitneyu(women_scores_filt, men_scores_filt, alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Interpret the result
alpha = 0.10
if p < alpha:
    print('Reject the null hypothesis (women score more than men)')
    r = 'reject'
else:
    print('Fail to reject the null hypothesis (no significant difference between women and men score)')
    r = 'fail to reject'


# In[ ]:


# The p-value and the result of the test must be stored in a dictionary called result_dict in the form:
# result_dict = {"p_val": p_val, "result": result}
# where p_val is the p-value and result is either the string "fail to reject" or "reject", depending on the result of the test.


# In[84]:


result_dict = {"p_val": p, "result": r}
result_dict


# In[ ]:





# In[ ]:




