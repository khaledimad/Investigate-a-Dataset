
# coding: utf-8

# # Project: Investigate a Dataset (Movie Database)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This Project will be focused on investigating The Movie Database (TMDb) which contains various information about movies such as their genres, runtime, budget etc. We will be investigating the following key questions:
# 
# 1) Which genres typically had a higher voting average?<br>
# 2) Which production companies had the highest spending budgets? How does their revenue compare to the overall database?<br>
# 3) What is the relationship between the movie budgets and their yielded **profits**?<br>
# 4) Which genres have been the most popular over the years?<br>
# 5) How much has the movie industry grown over the years? Are the runtimes generally increasing or decreasing with time?
# 

# ### Import Libraries
# First, we begin by importing all neccessary libraries to complete the project which include: *Pandas*, *NumPy*, *Seaborn*, *Matplotlib* and we later import the *csv file* and assign it to a dataframe **df**
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('tmdb-movies.csv')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# Now that the csv file has been imported, we can now begin exploring the data in order to fine tune it for our analysis practice.
# 
# ### General Properties
# 
# The first glance at the first five rows of the entire dataset

# In[2]:


df.head(5)


# With "info", we are able to examine the dataset types associated with each column and the total number of non-null values per column.

# In[3]:


df.info()


# With "describe", we are able to further examine basic statistics of the dataset such as the minimum, maximum values and standard deviations.

# In[4]:


df.describe()


# ### Data Cleaning
# 
# In this part, we will be fine-tuning the existing dataset in the following ways:
# 
# 1) Deleting columns which are not neccessary for the project exercise.<br>
# 2) Dropping any duplicated rows.<br>
# 3) Tweaking odd values shown in the columns under study.

# 1) Start with deleting all unneccessary columns which will not be used for our analysis
# 
# 

# In[5]:


df.drop(['id', 'imdb_id', 'popularity', 'cast', 'homepage', 'tagline', 'overview', 'release_date', 'budget_adj', 'revenue_adj'], axis=1, inplace=True)


# Confirm results >>>

# In[6]:


df.info()


# 2) Drop rows which are repeated for all columns

# In[7]:


sum(df.duplicated())
df.drop_duplicates(inplace = True)


# Confirm results >>>

# In[8]:


df.info()


# 3) Tweaking the values
# 
# While using the describe feature during the wrangling process, it was noticed that some of the values under study were showing odd values. For example, the budget, revenue and runtime columns had a **min** value of 0 which is an unusual occurence. Hence, next exercise will be to tweak those values either by ommitting the entire rows from the dataset or by converting them to null values whichever is more suitable.

# **Check 1:** We will first need to locate the rows which are showing a <u>*budget*</u> equivalent to 0. We filter to show the first row alone.

# In[9]:


df_check = df.loc[(df[['budget']] == 0).all(axis=1)]

df_check.head(1)


# According to the above, the movie **Mr.Holmes** is showing a budget of 0; however, as per IMDB it had an estimated budget of $10,000,000. Thus, making this value unreliable.<br><br>
# 
# **Check 2:** Next we will repeat the same exercise with <u>*revenue*</u>. Instead of showing the first row, we can show the last row by replacing *head* with *tail*.

# In[10]:


df_check2 = df.loc[(df[['revenue']] == 0).all(axis=1)]

df_check2.tail(1)


# According to above, the movie **Manor: The Hands of Fate** is showing a revenue of zero; however, as per IMDB it had a US Gross of $26,285,544. Thus, making this value unreliable.<br><br>
# 
# **Check 3:** Next we will repeat the same exercise with <u>*Runtime*</u>. 
# 

# In[11]:


df_check3 = df.loc[(df[['runtime']] == 0).all(axis=1)]

df_check3.head(1)


# According to above, the movie **Mythica: The Necromancer** is showing a runtime of 0 minutes; whereas, on IMDB the movie has a runtime of 1 hr 33mins. Thus, making this value unreliable.<br><br>
# 
# Based on the results above, we notice that the values displayed under these columns are incorrect and should not be considered. It is best to replace zero values with null values in order to maintain record of remaining columns which may come to use to analyze remaining data in other questions such as genres, keywords, etc.

# In[12]:


cols = ["budget", "revenue", "runtime"]
df[cols] = df[cols].replace({0:np.nan, 0:np.nan})
df.describe()


# Confirm results >>>

# In[13]:


df.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Now that we have a clean dataset, we are able to move on further with investigating the dataset. We will first begin with identifying a relationship between the genres and the movie voting average. This way, we will be able to further understand which genres usually yield the higher ratings.
# 
# 
# ### Research Question 1 - Which genres typically had a higher voting average?

# First we should check the number of unique genres

# In[14]:


df.genres.nunique()


# Due to genres column containing multiple values separated by pipe (|), there are way too many unique points. To resolve this, we will need to segregate the values under genre by splitting them from the pipes and creating a new unique row for each genre.

# In[15]:


genres_split = df['genres'].str.split('|').apply(pd.Series, 1).stack()
genres_split.index = genres_split.index.droplevel(-1)
genres_split.name = 'genres'

del df['genres']

df_genres = df.join(genres_split)

df_genres.head()


# Let us check the number of unique genres now after the split.

# In[16]:


df_genres.genres.nunique()


# Now we have a more realistic number of genres of 20. Let us check the sum of vote counts per genre now.

# In[17]:


vc_sum = df_genres.groupby(['genres']).sum().vote_count

vc_sum


# We can now check the average number of votes across each genre

# In[18]:


df_genres.groupby(['genres']).mean()


# Above information is **not very reliable** as it considers an equal weight for the vote average per movie although there are different number of votes distributed for each movie. Hence, it is best to calculate the sum of the product of the vote counts and vote average and divide it by the total number of votes per genre for a fair measure as per below equation:<br>
# 
# \begin{equation*}
# average\_vote = \frac{\sum_{k=1}^n vote\_average * vote\_count }{sum\_of\_votes}
# \end{equation*}<br>
# 
# 
# Let us begin by adding a new column for product of vote_count and vote_average.

# In[19]:


df_genres['vote_product'] = df_genres['vote_count']*df_genres['vote_average']


# Next we divide the sum of products by the total number of votes.

# In[20]:


sum_of_products = df_genres.groupby(['genres']).sum().vote_product

sum_of_votes = df_genres.groupby(['genres']).sum().vote_count

average_vote = sum_of_products / sum_of_votes

average_vote.sort_index()


# Now that we have a proper average for each genre, we can proceed with producing a visual representation to show the distribution of voting average across different genres. 
# 
# For a clearer pattern, we will sort values shown above from largest to smallest.

# In[21]:


df_genre_vote = pd.DataFrame({'genres': average_vote.index, 'average_vote': average_vote.values})
df_genre_vote

df_genre_vote = df_genre_vote.sort_values(by='average_vote', ascending=False)

df_genre_vote 

f, ax = plt.subplots(figsize = (8, 5))
sns.barplot(x = 'average_vote', y = 'genres', data = df_genre_vote)
ax.set_title('Average Votes across Genres')
ax.set_xlabel('Average Vote')
ax.set_ylabel('Genres')


# According to the above barplot, documentary, war and history genres have the highest average votes. However, let us examine the number of votes for these genres with respect to the total number of votes for all genres.<br>
# 
# To do this, we will need to calculate the percentage of the vote counts per genre as per the following equation: 
# 
# \begin{equation*}
# ratio\_perc = \frac{100 * vc\_sum}{vc\_sum\_total}
# \end{equation*}<br>
# 
# Once the results are produced, we can plot it on a pie chart to show the relative number of votes per genres.

# In[22]:


ratio = vc_sum / vc_sum.sum()

ratio_perc = ratio*100

explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.4,0.5,0.5,0.4,0.1,0.1,0.1,0.1,0.1,0.1)

ratio_perc

ratio_perc = ratio_perc.plot.pie(y='ratio_perc', explode = explode, fontsize=10, figsize=(8, 8), autopct='%1.0f%%', labeldistance=1.05);
plt.title("Percentage of Vote Counts across Genres");
ratio_perc.set_ylabel('');


# According to the pie chart, the documentary, war and history genres comprise a very small part of the total number of votes deeming the ratings unreliable. Instead, we ought to consider genres with a relatively high number of vote counts such as action and drama.<br>
# 
# From the previous barplot we notice that the drama genre comes after documentary, war and history. However, since it has a reliable number of vote counts, we can consider that drama has the highest movie ratings relative to the popular vote count.

# ### Research Question 2  - Which production companies have the highest spending budgets? How does their revenue compare to the overall database?

# Similar to the genres criteria, the production companies also have multiple values listed under the same cell separated by pipe (|); hence, the first exercise will be to split the cells to individual unique values.

# In[23]:


prod_comp_split = df['production_companies'].str.split('|').apply(pd.Series, 1).stack()
prod_comp_split.index = prod_comp_split.index.droplevel(-1)
prod_comp_split.name = 'production_companies'

del df['production_companies']

df_prod_comp = df.join(prod_comp_split)


# Confirm results >>>

# In[24]:


df_prod_comp.info()


# In[25]:


df_prod_comp.nunique().production_companies


# After splitting the production companies, we find out that there is a total of 4448 unique production companies in this dataset.

# Now we need to calculate the overall budget spent per production company then divide it over the cumulative budget for all movies in order to calculate the ratio of budget spent per production company to the overall budget.
# 
# Below equation will be used:
# 
# 
# \begin{equation*}
# comp\_budget\_ratio\_perc = \frac{comp\_overall\_budget * 100 }{overall\_budget}
# \end{equation*}<br>
# 
# We will then filter the results to obtain the top 3 spending production companies.

# In[26]:


comp_overall_budget = df_prod_comp.groupby(['production_companies']).sum().budget

overall_budget = df_prod_comp.groupby(['production_companies']).sum().budget.sum()

comp_budget_ratio_perc = comp_overall_budget * 100 / overall_budget

top_3_budget_comp = comp_budget_ratio_perc.nlargest(3)

top_3_budget_comp


# As per the results, **Warner Bros.**, **Universal Pictures** and **Paramount Pictures** are the top spending production companies with an overall spending budget in excess of 10% of the entire budget for 4448 production companies in the timeframe under study.
# 
# However, do <u>big spenders</u> neccessarily yield <u>greater revenues</u>???<br>
# 
# Let's check it out, using the same sequence of steps from the previous exercise.

# In[27]:


comp_overall_revenue = df_prod_comp.groupby(['production_companies']).sum().revenue

overall_revenue = df_prod_comp.groupby(['production_companies']).sum().revenue.sum()

comp_revenue_ratio_perc = comp_overall_revenue * 100 / overall_revenue

top_3_revenue_comp = comp_revenue_ratio_perc.nlargest(3)

top_3_revenue_comp


# Below is a bar chart to better illustrate the top 3 spending production companies which also correspond to the top 3 revenue making companies.

# In[28]:


budg_rev = pd.concat([top_3_budget_comp, top_3_revenue_comp], axis=1)

budg_rev

fig = plt.figure() 

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.3

budg_rev.budget.plot(kind='bar', color='red', ax=ax, width=width, position=1)
budg_rev.revenue.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Budget %');
ax2.set_ylabel('Revenue %');

ax.set_title('Top 3 Production Companies in terms of Budget and Revenue');


# According to the above results, the top 3 revenue gainers as a whole are also **Warner Bros.**, **Universal Pictures** and **Paramount Pictures** in the same order. The revenue of these 3 companies comprise slightly over 11% of the revenue of 4448 companies.
# 
# Does this however insinuate higher profits and profit percentages?? In the next exercise, we will check if higher budget movies yield higher **profit** percentages.
# 
# 

# ### Research Question 3  - Do higher budgets yield higher <u>PROFITS</u>?

# Lets begin by adding a profits column to our previous dataframe.

# In[29]:


df_prod_comp['profit'] = df_prod_comp['revenue']-df_prod_comp['budget']

df_prod_comp.head()


# Now lets look at profit as a percentage of budget.

# In[30]:


df_prod_comp['profit_perc'] = df_prod_comp['profit'] *100 / df_prod_comp['budget']

df_prod_comp.info()


# Now let us draw a relationship between the budget and profit percentage using a scatter plot.

# In[31]:


df_prod_comp.plot(x='budget', y = 'profit_perc', title = 'Profit percentage vs. Budget', kind='scatter');


# Looking at above plot, it seems that profit percentage is excessively high at some points. Hence, let us examine these critical points in more detail.
# 
# We will check rows where profit percentage is higher than 10000.

# In[32]:


df_prod_comp.query('profit_perc > 10000')


# Looking through the values, it is noticed that many movies are showing very low budgets < 500 which is also not reflecting the true values as per IMDB. Hence, it is best to filter out these values.
# 
# Let us try plotting again after applying the filter.

# In[33]:


df_prod_comp = df_prod_comp.loc[(df[['budget']] > 500).all(axis=1)]

df_prod_comp.plot(x='budget', y = 'profit_perc', title = 'Profit percentage vs. Budget', kind='scatter');


# Looking at the above graph after applying filters to unrealistic data, it seems that the profit percentage is highest where the budget is minimal.
# 
# Let us draw a relationship of the top production companies in terms of profit percentage now.

# In[34]:


prof_perc = df_prod_comp.groupby(['production_companies']).max().profit_perc.nlargest(10)

prof_perc = pd.DataFrame({'production_companies': prof_perc.index, 'prof_perc': prof_perc.values})

prof_perc = prof_perc.sort_values(by='prof_perc', ascending=False)

f, ax = plt.subplots(figsize = (9, 7))
sns.barplot(x = 'prof_perc', y = 'production_companies', data = prof_perc)
ax.set_title('Most profit generating production companies relative to budget');
ax.set_xlabel('Profit Percentage');
ax.set_ylabel('Production Companies');


# Looking at the top 10 profit generating production companies; we notice that none of them are similar to the top 3 from the previous exercise!
# 
# Hence, even though the top spending production companies made the largest revenues; that did not neccessarily mean they earned more for every dollar spent. Instead the top two profitable movies were **Paranormal Activity** and **The Blair Witch Project** which had budgets of *15000 USD* and *25000 USD* respectively. 

# ### Research Question 4  - Which genres have been the most popular over the years?

# To begin this exercise, we will need to use the genre dataset which was previously split into twenty unique genres.
# 
# Next, we will need to calculate the genre percentage over the timeframe of the dataset using the following equation:
# 
# \begin{equation*}
# genre\_perc = \frac{genre\_qty * 100 }{total\_genre}
# \end{equation*}<br>

# In[35]:


genre_qty = df_genres.groupby(['genres']).size()

total_genre = genre_qty.sum()

genre_perc = genre_qty * 100 / total_genre 

genre_perc


# Now let us better visualize this data using a bar plot.

# In[36]:


genre_perc = pd.DataFrame({'genres': genre_perc.index, 'genre_perc': genre_perc.values})

genre_perc = genre_perc.sort_values(by='genre_perc', ascending=False)

f, ax = plt.subplots(figsize = (9, 7))
sns.barplot(x = 'genre_perc', y = 'genres', data = genre_perc);
ax.set_title('Popularity across Genres from 1960 - 2015');
ax.set_xlabel('% from overall');
ax.set_ylabel('Genres');


# According to the results above, Drama contemplates the largest % of the overall movie genres.

# ### Research Question 5  - How much has the movie industry grown over the years? Are the runtimes generally increasing or decreasing with time?

# For the following exercise, we will draw two scatter plots showing the evolution of the movie industry over the past 55 years. Afterwards, we will examine how the runtimes have varied over the years.

# <u>Movie count over time:</u>

# In[37]:


trend_year = df.groupby(['release_year']).size()

trend_year = pd.DataFrame({'Year': trend_year.index, 'Movie_Count': trend_year.values})

trend_year.plot(x='Year', y = 'Movie_Count', title = 'Movie Count over the Years', kind='scatter');


# According to above, the movie industry has been growing exponentially over the past half a century with an immense growth starting from the 1990s.
# 
# <u>Movie runtime over time:</u>

# In[38]:


trend_year2 = df.groupby(['release_year']).mean().runtime

trend_year2

trend_year2 = pd.DataFrame({'Year': trend_year2.index, 'runtime': trend_year2.values})

trend_year2.plot(x='Year', y = 'runtime', title = 'Runtime over the years', kind='scatter');


# According to the above plot, average movie runtimes are generally decreasing with time starting from the 120 minute average down to around 95 today.

# <a id='conclusions'></a>
# ## Conclusions
# 
# In conclusion, the purpose of this project was to investigate The Movie Database (TMDB) in order to explore the relationship between certain parameters and the evolution of the movie industry over the years. The data accumulated was merely an estimation due to some missing links, for example several parameters were not displaying proper information and would infact be required for a more accurate result.
# 
# Using the neccessary functions and visualization tools, we were able to notice that the movies in the drama genre are the most popular in terms of quantity and vote count. Also, they usually yield higher voting averages through the popular vote count.
# 
# It was also noticed that the top 3 spending studios **Warner Bros.**, **Universal Pictures** and **Paramount Pictures** also returned the highest revenues percentage-wise. However, their profits in respect to the budget spent was not among the highest. Instead, it was noticed that low budget movies usually yield higher profits.
# 
# Finally, we can see that the movie industry is growing at an exponential rate where the number of movies per year has increased from about 30 per year to about 700 per year which was more than 20 fold. Also the average movie runtime has considerably decreased by 20% from around 120 minutes to 95 minutes today.

# In[40]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

