# Investigate-a-Dataset
Udacity Data Analyst Degree - Project III

## Overview
In this project, I will go through the data analysis process to investigate The Movie Database (TMDb) which contains various information about movies such as their genres, runtime, budget etc. 

## What Software Do I Need?

To complete this project, I will be using Python plus the following libraries:

- Pandas and NumPy (speeds up data analysis code)
- Matplotlib (produce plots)

## Asking Questions

I chose to investigate the following questions from the dataset:

1) Which genres typically had a higher voting average?
2) Which production companies had the highest spending budgets? How does their revenue compare to the overall database?
3) What is the relationship between the movie budgets and their yielded profits?
4) Which genres have been the most popular over the years?
5) How much has the movie industry grown over the years? Are the runtimes generally increasing or decreasing with time?

## Data Wrangling

In this part, I fine-tuned the existing dataset by:

1) Deleting unnecessary columns
2) Dropping duplicated rows
3) Tweaking illogical values

## Exploratory Data Analysis

Once I had a clean dataset, I was able to move on further with investigating the dataset. To answer the questions, pandas' series and DataFrame objects were used to access data more conveniently. Once the questions were answered, they were plotted using Matplotlib for clear visuals of the results as shown below:

#### <b>Voting average across genres:</b>
![screen shot 2018-09-30 at 9 11 36 pm](https://user-images.githubusercontent.com/43564654/46260513-0636fa80-c4f8-11e8-834d-85985ac4c902.png)


#### <b>Highest budget production companies and respective revenues:</b>

![screen shot 2018-09-30 at 9 21 28 pm](https://user-images.githubusercontent.com/43564654/46260524-28307d00-c4f8-11e8-9ced-218a9104a11c.png)


#### <b>Budgets versus profits:</b>


![screen shot 2018-09-30 at 9 21 51 pm](https://user-images.githubusercontent.com/43564654/46260535-48f8d280-c4f8-11e8-8620-430695d2c4be.png)


#### <b>Genre popularity over time:</b>

![screen shot 2018-09-30 at 9 22 08 pm](https://user-images.githubusercontent.com/43564654/46260537-544bfe00-c4f8-11e8-8cf9-63836caf1a75.png)


#### <b>Movie count over time:</b>

![screen shot 2018-09-30 at 9 22 39 pm](https://user-images.githubusercontent.com/43564654/46260543-5f9f2980-c4f8-11e8-8593-b076c1f333a5.png)

#### <b>Movie runtime over the years:</b>


![screen shot 2018-09-30 at 9 22 44 pm](https://user-images.githubusercontent.com/43564654/46260544-6037c000-c4f8-11e8-8f34-72b87e4ab923.png)

## Drawings Conclusions

Using the neccessary functions and visualization tools, it may be noticed that the movies in the drama genre are the most popular. 

It was also noticed that the top 3 spending studios also returned the highest revenues percentage-wise. However, their profits in respect to the budget spent was not among the highest. Instead, it was noticed that low budget movies usually yield higher profits.

It is also observed that the movie industry is growing at an exponential rate where the number of movies per year has increased from about 30 per year to about 700 per year. Also the average movie runtime has considerably decreased by 20% from around 120 minutes to 95 minutes today.
