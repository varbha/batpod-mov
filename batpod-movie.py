
# coding: utf-8

# <p style="font-family: Arial; font-size:3.75em;color:purple; font-style:bold"><br>
# Pandas</p><br>

# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Import Libraries
# </p>

# In[2]:

import pandas as pd


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">


# In[4]:

ser = pd.Series([100, 'foo', 300, 'bar', 500], index= ['tom', 'bob', 'nancy', 'dan', 'eric'])


# In[5]:

ser


# In[6]:

ser.index


# In[7]:

ser.loc[['nancy','bob']]


# In[8]:

ser[[4, 3, 1]]


# In[9]:

ser.iloc[2]


# In[10]:

'bob' in ser


# In[11]:

ser


# In[12]:

ser * 2


# In[13]:

ser[['nancy', 'eric']] ** 2


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# pandas DataFrame</p>
# 
# *pandas DataFrame* is a 2-dimensional labeled data structure.

# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Create DataFrame from dictionary of Python Series</p>

# In[5]:

d = {'one' : pd.Series([100., 200., 300.], index=['apple', 'ball', 'clock']),
     'two' : pd.Series([111., 222., 333., 4444.], index=['apple', 'ball', 'cerill', 'dancy'])}


# In[17]:

df = pd.DataFrame(d)
df


# In[16]:

df.index


# In[18]:

df.columns


# In[19]:

pd.DataFrame(d, index=['dancy', 'ball', 'apple'])


# In[20]:

pd.DataFrame(d, index=['dancy', 'ball', 'apple'], columns=['two', 'five'])


# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Create DataFrame from list of Python dictionaries</p>

# In[21]:

data = [{'alex': 1, 'joe': 2}, {'ema': 5, 'dora': 10, 'alice': 20}]


# In[22]:

pd.DataFrame(data)


# In[23]:

pd.DataFrame(data, index=['orange', 'red'])


# In[24]:

pd.DataFrame(data, columns=['joe', 'dora','alice'])


# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Basic DataFrame operations</p>

# In[25]:

df


# In[26]:

df['one']


# In[27]:

df['three'] = df['one'] * df['two']
df


# In[28]:

df['flag'] = df['one'] > 250
df


# In[29]:

three = df.pop('three')


# In[30]:

three


# In[31]:

df


# In[32]:

del df['two']


# In[33]:

df


# In[34]:

df.insert(2, 'copy_of_one', df['one'])
df


# In[35]:

df['one_upper_half'] = df['one'][:2]
df


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# Case Study: Movie Data Analysis</p>
# <br>This notebook uses a dataset from the MovieLens website. We will describe the dataset further as we explore with it using *pandas*. 
# 
# ## Download the Dataset
# 
# Please note that **you will need to download the dataset**. Although the video for this notebook says that the data is in your folder, the folder turned out to be too large to fit on the edX platform due to size constraints.
# 
# Here are the links to the data source and location:
# * **Data Source:** MovieLens web site (filename: ml-20m.zip)
# * **Location:** https://grouplens.org/datasets/movielens/
# 
# Once the download completes, please make sure the data files are in a directory called *movielens* in your *Week-3-pandas* folder. 
# 
# Let us look at the files in this dataset using the UNIX command ls.
# 

# In[3]:

# Note: Adjust the name of the folder to match your local directory
pwd
get_ipython().system('ls ./week-4-pandas/week-4-pandas/ml-20m')


# In[ ]:

get_ipython().system('cat ./movielens/movies.csv | wc -l')


# In[ ]:

get_ipython().system('head -5 ./movielens/ratings.csv')


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# Use Pandas to Read the Dataset<br>
# </p>
# <br>
# In this notebook, we will be using three CSV files:
# * **ratings.csv :** *userId*,*movieId*,*rating*, *timestamp*
# * **tags.csv :** *userId*,*movieId*, *tag*, *timestamp*
# * **movies.csv :** *movieId*, *title*, *genres* <br>
# 
# Using the *read_csv* function in pandas, we will ingest these three files.

# In[4]:

movies = pd.read_csv('./ml-20m/movies.csv', sep=',')
print(type(movies))
movies.head(15)


# In[6]:

# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

tags = pd.read_csv('./ml-20m/tags.csv', sep=',')
tags.head()


# In[7]:

ratings = pd.read_csv('./ml-20m/ratings.csv', sep=',', parse_dates=['timestamp'])
ratings.head()


# In[8]:

# For current analysis, we will remove timestamp (we will come back to it!)

del ratings['timestamp']
del tags['timestamp']


# <h1 style="font-size:2em;color:#2467C0">Data Structures </h1>

# <h1 style="font-size:1.5em;color:#2467C0">Series</h1>

# In[9]:

#Extract 0th row: notice that it is infact a Series

row_0 = tags.iloc[0]
type(row_0)


# In[10]:

print(row_0)


# In[11]:

row_0.index


# In[12]:

row_0['userId']


# In[13]:

'rating' in row_0


# In[14]:

row_0.name


# In[15]:

row_0 = row_0.rename('first_row')
row_0.name


# <h1 style="font-size:1.5em;color:#2467C0">DataFrames </h1>

# In[16]:

tags.head()


# In[17]:

tags.index


# In[18]:

tags.columns


# In[19]:

# Extract row 0, 11, 2000 from DataFrame

tags.iloc[ [0,11,2000] ]


# <h1 style="font-size:2em;color:#2467C0">Descriptive Statistics</h1>
# 
# Let's look how the ratings are distributed! 

# In[20]:

ratings.head(5)


# In[21]:

ratings['movieId'].describe()


# In[22]:

ratings.describe()


# In[23]:

ratings['rating'].mean()


# In[24]:

ratings.mean()


# In[25]:

ratings['rating'].min()


# In[26]:

ratings['rating'].max()


# In[27]:

ratings['rating'].std()


# In[28]:

ratings['rating'].mode()


# In[29]:

ratings.corr()


# In[30]:

filter_1 = ratings['rating'] > 5
print(filter_1)
filter_1.any()


# In[31]:

filter_2 = ratings['rating'] > 0

filter_2.all()


# <h1 style="font-size:2em;color:#2467C0">Data Cleaning: Handling Missing Data</h1>

# In[32]:

movies.shape


# In[33]:

#is any row NULL ?

movies.isnull().any()


# Thats nice ! No NULL values !

# In[34]:

ratings.shape


# In[35]:

#is any row NULL ?

ratings.isnull().any()


# Thats nice ! No NULL values !

# In[36]:

tags.shape


# In[37]:

#is any row NULL ?

type((tags.isnull().any()))
print (tags.isnull().any())


# We have some tags which are NULL.

# In[38]:

tags = tags.dropna()


# In[39]:

#Check again: is any row NULL ?

tags.isnull().any()


# In[40]:

tags.shape


# In[41]:

movies


# Thats nice ! No NULL values ! Notice the number of lines have reduced.

# <h1 style="font-size:2em;color:#2467C0">Data Visualization</h1>

# In[42]:

get_ipython().magic('matplotlib inline')

ratings.hist(column='rating', figsize=(15,5))


# In[43]:

ratings.boxplot(column='rating', figsize=(15,20))


# <h1 style="font-size:2em;color:#2467C0">Slicing Out Columns</h1>
#  

# In[44]:

tags['tag'].head()


# In[45]:

movies[['title','genres']].head()


# In[46]:

ratings[ratings['userId']==11]
print("---")
ratings[-10:]


# In[47]:

tag_counts = tags['tag'].value_counts()
tag_counts[:10]


# In[48]:

get_ipython().magic('matplotlib inline')
tag_counts[-10:].plot(kind='bar', figsize=(15,10))


# <h1 style="font-size:2em;color:#2467C0">Filters for Selecting Rows</h1>

# In[49]:

is_highly_rated = ratings['rating'] >= 4.0

ratings[is_highly_rated][-30:]


# In[50]:

is_animation = movies['genres'].str.contains('Animation')

movies[is_animation][5:15]


# In[51]:

movies[is_animation].head(15)


# <h1 style="font-size:2em;color:#2467C0">Group By and Aggregate </h1>

# In[52]:

ratings_count = ratings[['movieId','rating']].groupby('rating').count()

ratings_count


# In[53]:

average_rating = ratings[['movieId','rating']].groupby('movieId').mean()
average_rating.head(10)


# In[54]:

movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.head()


# In[55]:

movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.tail()


# <h1 style="font-size:2em;color:#2467C0">Merge Dataframes</h1>

# In[56]:

tags.head()


# In[57]:

movies.head()


# In[58]:

t = movies.merge(tags, on='movieId', how='inner')
t.head()




# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# 
# Combine aggreagation, merging, and filters 
# </p>

# In[59]:

avg_ratings = ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
avg_ratings.head()


# In[60]:

box_office = movies.merge(avg_ratings, on='movieId', how='inner')
box_office.tail()


# In[61]:

is_highly_rated = box_office['rating'] >= 4.0

box_office[is_highly_rated][-5:]


# In[62]:

is_comedy = box_office['genres'].str.contains('Comedy')

box_office[is_comedy][:5]


# In[63]:

box_office[is_comedy & is_highly_rated][-5:]


# <h1 style="font-size:2em;color:#2467C0">Vectorized String Operations</h1>
# 

# In[64]:

movies.head()


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Split 'genres' into multiple columns
# 
# <br> </p>

# In[65]:

movie_genres = movies['genres'].str.split('|', expand=True)


# In[66]:

movie_genres[:10]


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Add a new column for comedy genre flag
# 
# <br> </p>

# In[67]:

movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')


# In[68]:

movie_genres[:10]


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Extract year from title e.g. (1995)
# 
# <br> </p>

# In[69]:

movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)


# In[70]:

movies.tail()


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>

# <h1 style="font-size:2em;color:#2467C0">Parsing Timestamps</h1>



# In[73]:

tags = pd.read_csv('./ml-20m/tags.csv', sep=',')


# In[74]:

tags.dtypes


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Unix time / POSIX time / epoch time records 
# time in seconds <br> since midnight Coordinated Universal Time (UTC) of January 1, 1970
# </p>

# In[75]:

tags.head(5)


# In[76]:

tags['parsed_time'] = pd.to_datetime(tags['timestamp'], unit='s')


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Data Type datetime64[ns] maps to either <M8[ns] or >M8[ns] depending on the hardware
# 
# </p>

# In[77]:


tags['parsed_time'].dtype


# In[78]:

tags.head(2)


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Selecting rows based on timestamps
# </p>

# In[79]:

greater_than_t = tags['parsed_time'] > '2015-02-01'

selected_rows = tags[greater_than_t]

tags.shape, selected_rows.shape


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Sorting the table using the timestamps
# </p>

# In[80]:

tags.sort_values(by='parsed_time', ascending=True)[:10]


# <h1 style="font-size:2em;color:#2467C0">Average Movie Ratings over Time </h1>
# ## Are Movie ratings related to the year of launch?

# In[81]:

average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()
average_rating.tail()


# In[82]:

joined = movies.merge(average_rating, on='movieId', how='inner')
joined.head()
joined.corr()


# In[83]:

yearly_average = joined[['year','rating']].groupby('year', as_index=False).mean()
yearly_average[:10]


# In[84]:

yearly_average[-20:].plot(x='year', y='rating', figsize=(15,10), grid=True)


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Do some years look better for the boxoffice movies than others? <br><br>
# 
# Does any data point seem like an outlier in some sense?
# 
# </p>

# In[ ]:



