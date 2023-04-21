


# How can we maximise profit as an airbnb host?


## Contributors:
 - Isaac Chun Jun Heng U2221389B
 - J'sen Ong Jia Xuan U2220457J
 - Tang Teck Meng U2221809C

## Code located in:

1. (Master File) [AirbnbAnalysis.ipynb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/AirbnbAnalysis.ipynb)
2. [EDA_First25.ipynb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_First25.ipynb)
3. [EDA_Middle25.ipynb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Middle25.ipynb)
4. [EDA_Last25.ipnyb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Last25.ipynb)
5. [Airbnb_EDA_Vizualization.ipnyb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_EDA_Visualization.ipynb)
6. [Airbnb_LinearRegression.ipnyb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_LinearRegression.ipynb)
7. [Airbnb_Machine_Learning.ipynb](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_Machine_Learning.ipynb)
## We are using Airbnb Singapore Dataset on 29 December 2022 and the source is located in http://insideairbnb.com/get-the-data/

# Content Section

1. What is Airbnb? (Introduction)
2. Introduction to Problem Statement and motivation
3. Libraries used
4. Explaining general utility functions that we have created
5. Dataset importing and cleaning of dataset
6. Exploratory Data Analysis
7. Machine Learning
8. Conclusion
9. Video Presentation
10. References

# What is Airbnb?

Airbnb, Inc. is an American San Francisco-based company operating an online marketplace for short-term homestays and experiences. The company acts as a broker and charges a commission from each booking. The company was founded in 2008 by Brian Chesky, Nathan Blecharczyk, and Joe Gebbia.

![enter image description here](https://mma.prnewswire.com/media/1121685/Airbnb_Logo.jpg)

## Introduction to Problem Statement and motivation

**Problem Statement: Identify the factors that could help maximise profit on Airbnb**
<b></b>
Motivation: Help Airbnb host get a better idea on how to maximize profits for their Airbnb listings.
Hypothesis: The number of amenities a listing provides will affect its price, the more the amenities, the higher the listing price and variables related to a listing's review will have positive correlation to listing's price.

## Libraries used:
NumPy: Library for Numeric Computations
Pandas: Data Acquisition and preparation
Matplotlib: Low-level library for Data Visualization
Seaborn: High-level library for Data Visualization
Wordcloud: Create word cloud
Folium: Visualization of data on map
Geopandas: Handle geojson data to generate chloropelth maps
Sklearn: Machine Learning

## Explaining General Utility functions that we have created
countOutliers(df) uses interquartile range method to count number of outliers

removeOutliers(df) uses the same principle as countOutliers to remove rows that have outliers

The purpose of the general utility function is to assist with our data cleaning process.

## Dataset importing and cleaning of Dataset
Our dataset is very large with 75 columns, hence, we first apply the pandas info method to get an overview of the columns that we are dealing with. By analysing the description of the columns, we have first exclude the columns that are obviously not relevant to our problem.
### Initial Visual Data Cleaning
We looked through the columns of our dataset and pick up the obviously redundant columns to drop such as the host identification number. This is shown in detail in the notebooks [EDA_First25](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_First25.ipynb), [EDA_Middle23](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Middle23.ipynb) and [EDA_last25](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Last25.ipynb).

The general utility functions we have made will assist in removing outliers in our data.

We will also be dropping Null values when nulls values do not mean anything in the column or that the number of null values are too small for the column. An example would be dropping null rows for host_location as an Airbnb listing should have a string that contains the location name that corresponds with it.

## Exploratory Data Analysis
We will look into univariate and bivariate EDAs concerning the interesting or important variables that we have identified.

We will be looking at the variables that are left after dropping the columns from our visual analysis.

This is divided into [EDA_First25](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_First25.ipynb), [EDA_Middle23](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Middle23.ipynb) and [EDA_Last25](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/EDA_Last25.ipynb). We conducted EDA on the remaining columns to confirm which columns are interesting or useful for our prediction.

### Initial EDA findings
We analysed the columns of our dataset to find out if the columns provide any meaningful insight for our project and also find out which columns would be useful for more in-depth analysis.
#### EDA_First25
host_location, host_location, host_is_superhost, host_verifications, host_response_rate have a very skewed data with 90% of the rows having the same output for each column respectively. Hence, we will drop these columns.
#### EDA_Middle25
The variable beds which describes the number of beds available is unbalanced as a huge majority of listings only has 1 bed available. Therefore, we will drop the beds column.
minimum_nights and maximum_nights will be used in our calculation of minimum and maximum profit a host can earn, but will not be directly used for prediction as a predictor.
#### EDA_Last25
The variables that are review related are skewed on the high end and is imbalanced, hence they shall be dropped.
We believe that license should be dropped as it is not necessary to have an Airbnb listing. 
A majority of host chose not to enable instant booking, which results in high False outputs for the column instant_bookable.
In conclusion, we can assume that with a higher reviews rating or score, it can help attract more guests and ultimately maximise our profit as an Airbnb host. However, it should be noted that there are many other factors that can influence booking rates and profitability.
### Uni-Variate EDA
For the remaining EDA, kindly refer to [Airbnb_Visualization](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_EDA_Visualization.ipynb) for detailed code and explanation. We included the overview of what we have observed from interesting/useful columns.
#### 1. host_response_time
This describes the average response time of the host to message![enter image description here](https://user-images.githubusercontent.com/105051750/233560668-bf7181c4-790d-48a3-b1e1-8b1326a47292.png)
We can observe that majority of the hosts has a response time of within an hour, with 855 hosts replying within an hour, about 21.95% of hosts reply within a day.

#### 2. neighbourhood_cleansed
The neighbourhood group as geocoded using the latitude and longitude against neighbourhoods as defined by open or public digital shapefiles.
![enter image description here](https://user-images.githubusercontent.com/105051750/233561559-16d087b1-4680-4a89-86b5-ce16fbbf3fb9.png)
We can see that Downtown Core has the highest amount of listings.
![enter image description here](https://user-images.githubusercontent.com/105051750/233561333-881ebbb9-234a-4164-b3ad-140064024929.png)

We can see that places like Kallang, Downtown Core and Outram has the highest proportion of houses as compared to the other areas.
#### 3. neighbourhood_group_cleansed
The neighbourhood group as geocoded using the latitude and longitude against neighbourhoods as defined by open or public digital shapefiles.
![enter image description here](https://user-images.githubusercontent.com/105051750/233561333-881ebbb9-234a-4164-b3ad-140064024929.png)
We can see that there are many property listings in the central region as compared to the other regions like North-East region or West region. This is probably due to the central region being closer as tourists attractions, so there is a higher amount of listings there. 

#### 4. property_type
Self-selected property type. Hotels and Bed and Breakfasts are described as such by their hosts in this field.
![enter image description here](https://user-images.githubusercontent.com/105051750/233562008-980bce20-09e1-42d0-a971-8e304b59ebcd.png)
For property type, there are mainly 3 types: Private, Entire and Shared. The rest are either hostels or other niche categories. We shall look into property type according to their categories for classification purposes.
![enter image description here](https://user-images.githubusercontent.com/105051750/233562147-35f206bb-92e5-4fdc-9a0f-00da47782cc9.png")
This representation makes it easier to analyse property_type, we will observe the impact of this variable with the listings price.

#### 5. accomodates
This is the maximum capacity of the listing.
![enter image description here](https://user-images.githubusercontent.com/105051750/233562351-ded51ad0-c916-431f-b2f0-eeea249b2942.png)
Generally, most listings have around 1-4 accomodates that can stay at a listing at a time.

#### 6.amenities
This describes the amenities present in the Airbnb listing
![enter image description here](https://user-images.githubusercontent.com/105051750/233562574-ee7320c4-b8d9-4e08-80fd-148f57c54492.png)
Many properties have amenities that allow long term stays, have air conditioning, as well as hot water, hair dryers and smoke alarm. This means that most properties that are on listing have these basic amenities.
![enter image description here](https://user-images.githubusercontent.com/105051750/233562445-8ce6292c-f7fc-43f8-afaf-bf2edf841815.png)
Looking at the number of amenities in a given Airbnb listing, we can see that this column has potential to being one of the predictors, with it being following a decent distribution, we will compare it to price later in our bivariate exploration.
#### 7. number_of_reviews
This describes the total number of reviews for each Airbnb listing
![enter image description here](https://user-images.githubusercontent.com/105051750/233562645-0df9d3b9-03be-4990-bb2b-e3497aaf9ef5.png)
We can observe that most of the listings have 0 reviews, so we can use this to check if the number of reviews affect the price of a listing in our bivariate exploration.
### Bi-Variate EDA
We will be using price as a response for simplicity.
#### 1. host_response_time vs price

 - Box plot of host_response_time against price
 ![enter image description here](https://user-images.githubusercontent.com/105051750/233562723-3bf03eac-aa20-491e-b272-bbad77164b7f.png)
 - There appears to be a positive correlation between response time and the price of the listing, suggesting that hosts who can reply within an hour tend to command higher prices compared to those who take a few days or more to respond.
#### 2. neighbourhood_cleansed vs price
We found that Downtown Core has the highest number of listings at 288. 
![enter image description here](https://user-images.githubusercontent.com/105051750/233563065-cadd98d8-eae9-4274-8087-bda1d76677fb.jpeg)
Listings at Southern Islands and Orchard have the highest median in price listings, which is corroborated by the fact that they are touristy areas. In comparison, Ang Mo Kio and Yishun have low Airbnb price listings relatively.
#### 3. neighbour_group_cleansed vs price
![enter image description here](https://user-images.githubusercontent.com/105051750/233562818-450da263-00f2-4d1b-942a-891cb59fda62.png)
 The central and western regions boast high median prices. However, Central has higher median for property listings than the West, which is within expectation. To add on, we noticed that Central region has the most outliers than any other categories, which is also within expectation as not all central listings are tourist spots, it is possible that the outliers are listings in tourist spots such as Orchard.
#### 4. property_type vs price
We observe the categories within property_type and surmised that they could be split mainly into four types of property: Private, entire, shared and hotel/hostels.
![enter image description here](https://user-images.githubusercontent.com/105051750/233563172-0f65db09-0a6e-4ef2-b2b5-1523b6cc2449.png)
We see that entire home apartment listings are generally more expensive than other listings, with the highest median as compared to other types of listings. Hotel/Hostels come in second, which is not surprising as hotels generally range around the high 200s range.
We can also observe that the amount of outliers for private rooms are very high, and this may be because if how the owner chooses to value his/her listing.
#### 5. Accomodates vs Price
![enter image description here](https://user-images.githubusercontent.com/105051750/233563258-7ef561c8-1df5-4a21-a501-ab3f9110439a.png)
From the correlation heat map, we found out that accomodates and price have a weak correlation.
#### 7. number_of_reviews vs price
![enter image description here](https://user-images.githubusercontent.com/105051750/233563311-cc828206-1a67-45d6-a650-63f308b04050.png)
From the correlation heat map, we found out that regardless of the outliers, the correlation between price and number_of_reviews are extremely weak and should not be used for further analysis.
### Insightful data
Here, we would like to cover interesting insights we have regarding a few variables. These variables may not be able to help with our prediction, but they can provide some interesting information regarding our problem.
#### 1. description
This describes the Airbnb listing itself.
Word Cloud of words in the Airbnb listing, the bigger the word, the more frequent the usage of that word is.
![enter image description here](https://user-images.githubusercontent.com/105051750/233563506-29d2ee4f-6bba-4341-bd10-9660a432b72e.png)
There is prominent use of words such as "fully furnished", "guest access", "space", "space", "located", "MRT station" and "walking distance", which reveals that there are high amount of listings that mention about walking distances near to MRT stations and also the interior of the house.

#### 2. neighbourhood_overview
This is the brief overview of the neighbourhood where the Airbnb listing is located.
![enter image description here](https://user-images.githubusercontent.com/105051750/233563595-6be1cca8-1d17-45fb-8e66-f78a56d7674f.png)
Words that has high usage include "MRT station", "mins walk", "bus stop", "Raffles Place", "Orchard Road", "Green Line", "Clarke Quay" and "food centre". It can be inferred that a higher proportion of listings are near the central region of Singapore, and are also within walking distance to the MRT station specifically East-West Line. (The Green MRT line)

#### 3. latitude and longitude
The latitude and longitude of the Airbnb listing
From this data, we can map out where each listing is on the Singaporean map.![enter image description here](https://user-images.githubusercontent.com/105051750/233563686-f82afb7b-dfed-423f-b4ca-c7a343411b09.png)
We can see that a large number of points are concentrated in the central region, as well as Kallang, Downtown Core and Outram.


## Machine Learning
We have chose regression as our Machine Learning model to help with our prediction. The notebook for this part is [Linear_Regression](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_LinearRegression.ipynb) and [Machine_Learning](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/blob/main/Airbnb_Machine_Learning.ipynb) which will include the code and more detailed information.
We split the dataset into train and test sets of 80:20 ratio for our machine learning model.
### Linear Regression Notebook
#### One hot encoding
We have decided to apply One-Hot Encoding to our categorical data to aid us in training such data using our machine learning models.
#### Linear Regression modelling
Firstly, we trained our model using all the features vs the price. We found that explained variance is relatively low and we have observed that there exist extreme predictions for certain features.
Next, we train each independent column against the price to gain some insight. We noticed that host_response_time, neighbourhood_cleansed, neighbourhood_group_cleansed, property_type, accommodates, number of amenities and number of reviews have a low explained variance and would not be helpful.
It is important to note that amenities vs price did yield signficantly better result with an explained variance of 0.5.
#### Ridge Regression
We decided to try Ridge Regression as it allows the analysis of data when multicollinearity is present and prevents overfitting of data. We noticed that Ridge regression yields very similar results to linear regression. 
#### Lasso Regression
We attempt lasso regression to see if it could yield better results as a linear model for our dataset. We also noticed similar results to linear regression.
#### Gradient Booster
Gradient Booster is the type of machine learning algorithm that is used to build predictive models using iterative improvement. We noted its ability to model non-linear relationships as our linear model have limited success for a few of our columns. It utilises weak learners such as decision tress to build stronger model.
#### XGBoost
XGBoost is a specific implementation of Gradient Booster that has higher performance and has improved model generalization capabilities. We noted that the results are very similar to that of Gradient Booster
#### Analysing the different models
![enter image description here](https://user-images.githubusercontent.com/105051750/233563833-706daa1c-e9e1-4b56-be72-d01c9c2803f8.png)
![enter image description here](https://user-images.githubusercontent.com/105051750/233563992-8c92e53c-4861-4ae9-99e7-f3eb7aa0230c.png)
![enter image description here](https://user-images.githubusercontent.com/105051750/233564052-650d2022-1063-4822-aae8-35e387465e8c.png)
For our linear models, they performed very similarly for each respective columns. We notice that a few of our columns might not have linear relationship with price. However, we have noticed that Lasso Regression did the best as the explained variance was the highest amongst the 3 models and that it had the lowest mean squared error of the 3 models.
Overall, Gradient Booster worked the best with an explained variance of 0.73 and a mean squared error of ~1900. This suggest that most of our data do indeed have non linear relationship with price. 
The most prominent column would be amenities, where it has a huge correlation with price. The second would be property_type, which means that these 2 columns have some impact on our price.
### Machine Learning Notebook
#### Multi-Variate K Means
We first fill in any missing values for each column with the median of that particular column. We chose to use the elbow method so that we will computed all the Sum of squared Errors(SSE) for each "k" and store into a list. We observe that the elbow point occurs between 2 to 3 of "Number of Clusters". We then apply another technique called silhouette coefficient to find the best k. We observe that 4 is the best option. Hence, we will split our data into 4 clusters. We then observe the graphs of each independent column against each of the 4 clusters. --To be Continued--



## Conclusion
--To be added-


## Video Presentation
--To be added--



# References
### Github repo : 
[https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/tree/main](https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/tree/main "https://github.com/isaacchunn/SC1015_MiniPrj_Airbnb/tree/main")  
  
### AIRBNB EDA exploration : 
[https://colab.research.google.com/drive/1Gh1p2MQXTThs134dRyLT6XqGNbGavm9J#scrollTo=tUY3F6v5qM96](https://colab.research.google.com/drive/1Gh1p2MQXTThs134dRyLT6XqGNbGavm9J#scrollTo=tUY3F6v5qM96 "https://colab.research.google.com/drive/1Gh1p2MQXTThs134dRyLT6XqGNbGavm9J#scrollTo=tUY3F6v5qM96")  
  
### Linear Regression : [https://colab.research.google.com/drive/1NJ7EAJzKxcokEkQ0YNJlczDjnhONOXlT#scrollTo=XWRNUW68KD1N](https://colab.research.google.com/drive/1NJ7EAJzKxcokEkQ0YNJlczDjnhONOXlT#scrollTo=XWRNUW68KD1N "https://colab.research.google.com/drive/1NJ7EAJzKxcokEkQ0YNJlczDjnhONOXlT#scrollTo=XWRNUW68KD1N")  
  
  
  
### RESEARCH resources: 
[https://www.kaggle.com/code/subhradeep88/airbnb-analysis-eda/notebook](https://www.kaggle.com/code/subhradeep88/airbnb-analysis-eda/notebook "https://www.kaggle.com/code/subhradeep88/airbnb-analysis-eda/notebook")  
[https://www.kaggle.com/code/ivanovskia1/maximize-value-of-airbnb-rental](https://www.kaggle.com/code/ivanovskia1/maximize-value-of-airbnb-rental "https://www.kaggle.com/code/ivanovskia1/maximize-value-of-airbnb-rental")  
[https://towardsdatascience.com/how-to-maximize-profits-on-airbnb-data-based-approach-for-hosts-beaf08f26941](https://towardsdatascience.com/how-to-maximize-profits-on-airbnb-data-based-approach-for-hosts-beaf08f26941 "https://towardsdatascience.com/how-to-maximize-profits-on-airbnb-data-based-approach-for-hosts-beaf08f26941")
![](blob:https://web.telegram.org/2fefbf5c-0041-422a-a761-b077bad5ac1a)
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
https://xgboost.readthedocs.io/en/stable/python/python_intro.html
https://medium.com/@jjosephmorrison/one-hot-encoding-to-set-up-categorical-features-for-linear-regression-6bac35661bb6
https://realpython.com/k-means-clustering-python/
https://machinelearningmastery.com/how-to-transform-data-to-fit-the-normal-distribution/
