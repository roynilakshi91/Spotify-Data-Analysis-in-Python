#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


# Loading the "tracks.csv" dataset
df_tracks = pd.read_csv("tracks.csv")
df_tracks.head()


# In[50]:


#Check for total null values for every columns
pd.isnull(df_tracks).sum()


# In[51]:


#Information of the dataframe
df_tracks.info()


# In[52]:


# Question 1: 10 least popular song in the Spotify dataset.

#Create an sorted dataframe to see least popular song 

sorted_df = df_tracks.sort_values("popularity", ascending = True).head(10)
sorted_df


# In[53]:


#Finding the statistics of the dataframe
df_tracks.describe().transpose()


# In[54]:


# Question 2: 10 most popular song whose popularity is greater than 90

most_popular = df_tracks.query('popularity>90', inplace = False).sort_values("popularity", ascending = False)
most_popular[:10]
#OR use head(10)


# In[55]:


#Set the index to release_date column in the original dataframe, inplace = True as we want the index to be changed
df_tracks.set_index("release_date", inplace = True)

#Changing it to datetime format 
df_tracks.index = pd.to_datetime(df_tracks.index)

#printing the head of the dataframe
df_tracks.head()


# In[56]:


# Question 3: Check the artist at the 18th row in the dataset

df_tracks[['artists']].iloc[18]


# In[57]:


# Convert the duration from milliseconds to seconds
df_tracks["duration"] = df_tracks["duration_ms"].apply(lambda x : round(x/1000))



# In[60]:


df_tracks.head()


# In[61]:


#Removing the duration_ms column
df_tracks.drop("duration_ms", inplace = True, axis = 1)


# In[62]:


df_tracks.head()


# In[63]:


# Creating a correlation maps between variables

corr_df = df_tracks.drop(["key", "mode", "explicit"],axis = 1).corr(method = "pearson")
plt.figure(figsize=(14,6))


# In[67]:


#Creating heatmap using seaborn

heatmap = sns.heatmap(corr_df,annot= True, fmt = ".1g", vmin = -1, vmax = 1, center = 0, cmap = "inferno", linewidths=1, linecolor = "Black")
heatmap.set_title("Correlation Heatmap between variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation = 90)


# In[69]:


# Creating a sample data frame
sample_df = df_tracks.sample(int(0.004 *len(df_tracks)))


# In[73]:


print(len(sample_df))


# In[74]:


# Creating a regression plot between loundness and energy
plt.figure(figsize=(10,6))
sns.regplot(data = sample_df,y = "loudness", x = "energy", color = "c").set(title = "loundness Vs energy correlation")


# In[76]:


# Creating a regression plot between popularity and acousticness
plt.figure(figsize=(10,6))
sns.regplot(data = sample_df,y = "popularity", x = "acousticness", color = "b").set(title = "popularity Vs acousticness correlation")


# In[77]:


# Create a new column called "year"

df_tracks["dates"] = df_tracks.index.get_level_values("release_date")
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year


# In[80]:


#pip install -- user seaborn == 0.11.0


# In[81]:


# Question 4: Total number of songs every year since 1942 that is streaming in spotify app
sns.displot(years, discrete = True, aspect = 2, height = 5, kind = "hist").set(title = "Total Number of songs per year")


# In[83]:


# Duration of songs over the year (barplot)

total_duration = df_tracks.duration
fig,ax = plt.subplots(figsize=(18,6))
fig = sns.barplot(x = years, y = total_duration, ax = ax , errwidth = False).set(title = "year Vs duration")
plt.xticks(rotation = 60)


# In[85]:


# Duration of songs over the year (lineplot)
total_duration = df_tracks.duration
sns.set_style(style= "whitegrid")
fig,ax = plt.subplots(figsize=(18,6))
fig = sns.lineplot(x = years, y = total_duration, ax = ax).set(title = "year Vs duration")
plt.xticks(rotation = 60)


# In[87]:


# Loading another dataset Spotify features

df_genre = pd.read_csv("Spotifyfeatures.csv")


# In[89]:


df_genre.head()


# In[90]:


# Question 5: Duration of the songs for different genre
plt.title("Duration of the Songs in Different Genre")
sns.color_palette("rocket", as_cmap = True)
sns.barplot(x = 'duration_ms', y = 'genre', data = df_genre)
plt.xlabel("duration in milli seconds")
plt.ylabel("genre")


# In[94]:


# Question 6: Top 5 genre by popularity

sns.set_style(style="darkgrid")
plt.figure(figsize=(10,5))
famous = df_genre.sort_values('popularity', ascending = False).head(10)
sns.barplot(y ="genre", x = 'popularity', data = famous). set(title = "TOP 5 Genre by Popularity")


# In[ ]:




