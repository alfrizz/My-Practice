#!/usr/bin/env python
# coding: utf-8

# ![NYC Skyline](nyc.jpg)
# 
# Welcome to New York City, one of the most-visited cities in the world. There are many Airbnb listings in New York City to meet the high demand for temporary lodging for travelers, which can be anywhere between a few nights to many months. In this project, we will take a closer look at the New York Airbnb market by combining data from multiple file types like `.csv`, `.tsv`, and `.xlsx`.
# 
# Recall that **CSV**, **TSV**, and **Excel** files are three common formats for storing data. 
# Three files containing data on 2019 Airbnb listings are available to you:
# 
# **data/airbnb_price.csv**
# This is a CSV file containing data on Airbnb listing prices and locations.
# - **`listing_id`**: unique identifier of listing
# - **`price`**: nightly listing price in USD
# - **`nbhood_full`**: name of borough and neighborhood where listing is located
# 
# **data/airbnb_room_type.xlsx**
# This is an Excel file containing data on Airbnb listing descriptions and room types.
# - **`listing_id`**: unique identifier of listing
# - **`description`**: listing description
# - **`room_type`**: Airbnb has three types of rooms: shared rooms, private rooms, and entire homes/apartments
# 
# **data/airbnb_last_review.tsv**
# This is a TSV file containing data on Airbnb host names and review dates.
# - **`listing_id`**: unique identifier of listing
# - **`host_name`**: name of listing host
# - **`last_review`**: date when the listing was last reviewed

# As a consultant working for a real estate start-up, you have collected Airbnb listing data from various sources to investigate the short-term rental market in New York. You'll analyze this data to provide insights on private rooms to the real estate company.
# 
# There are three files in the data folder: airbnb_price.csv, airbnb_room_type.xlsx, airbnb_last_review.tsv.

# In[1]:


# Import necessary packages
import pandas as pd
import numpy as np

# Begin coding here ...
# Use as many cells as you like


# In[6]:


airbnb_price = pd.read_csv('data//airbnb_price.csv')
airbnb_price


# In[9]:


airbnb_last_review = pd.read_csv('data//airbnb_last_review.tsv', sep='\t')
airbnb_last_review


# In[11]:


airbnb_room_type = pd.read_excel('data//airbnb_room_type.xlsx')
airbnb_room_type


# What are the dates of the earliest and most recent reviews? Store these values as two separate variables with your preferred names.

# In[12]:


airbnb_last_review.info()


# In[22]:


airbnb_last_review['last_review_dt'] = pd.to_datetime(airbnb_last_review.last_review)
airbnb_last_review


# In[26]:


earliest_review = min(airbnb_last_review['last_review_dt'])
most_recent_review = max(airbnb_last_review['last_review_dt'])

earliest_review, most_recent_review


# How many of the listings are private rooms? Save this into any variable.

# In[27]:


airbnb_room_type.room_type.unique()


# In[31]:


airbnb_room_type.room_type.str.lower().unique()


# In[33]:


airbnb_room_type['room_type'] = airbnb_room_type.room_type.str.lower()
airbnb_room_type.room_type.unique()


# In[42]:


num_private_rooms = airbnb_room_type.room_type.value_counts()['private room']
num_private_rooms


# What is the average listing price? Round to the nearest two decimal places and save into a variable.

# In[57]:


airbnb_price['price_int'] = airbnb_price.price.str.split(' ', n=1).str[0].astype(int)
airbnb_price


# In[60]:


aver_price_list = np.round(airbnb_price['price_int'].mean(), 2)
aver_price_list


# Combine the new variables into one DataFrame called review_dates with four columns in the following order: first_reviewed, last_reviewed, nb_private_rooms, and avg_price. The DataFrame should only contain one row of values.

# In[72]:


data = {
    'first_reviewed': [earliest_review],
    'last_reviewed': [most_recent_review],
    'nb_private_rooms': [num_private_rooms],
    'avg_price': [aver_price_list]
}

review_dates = pd.DataFrame(data)
review_dates


# In[ ]:




