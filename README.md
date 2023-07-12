## 🚀 About Me
I'm Ashutosh pursuing Masters in applied machine intelligence and as part of my data maining course I have utlized machine learning techiniques for Spotify song Popularity Analysis & Prediction. We have also craeted a pipeline which is an effective way to change and manipulate a lot of data. Pipelines are a series of ways to process data. With Pandas pipeline feature, we can string together different user-defined Python functions to make a pipeline for processing data.
Here we have created 3 small pipelines.



## Spotify Track Popularity – Analysis and Prediction

# Problem Statement
The objective of this project is to develop a machine learning model that accurately predicts the popularity of songs on the Spotify platform. Spotify is a popular music streaming service with a vast library of songs from various genres and artists. The popularity of a song is a crucial factor for both artists and users, as it determines the visibility, reach, and potential success of a song. I have conducted EDA and applied supervised and unsupervised ML methods to find the most popular songs and also we string together different user-defined Python functions to make a pipeline for processing data, I have created 3 small pipelines. I used over 600,000 Spotify tracks and 20 attributes in this project.

What is the overall trend and impact of notable events on the music industry?
Who are the most popular artists in the dataset?
What are the most popular songs in the dataset?
What types of machine learning models do well on this dataset?
What characteristics most clearly characterize a popular song?

## Data Source: - https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets

## 🔗 Links
![Code](spotify-track-popularity-analysis-and-prediction.ipynb)

##  Key Findings

1. What characteristics most clearly characterize a popular song?
<img width="652" alt="image" src="https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/4ae48a4b-0a48-4f34-88ee-22af31f32bef">
When we compared the most popular songs to all of the songs in the dataset, we discovered that songs with higher energy and danceability tend to become more popular. However, when compared to standard songs, such popular songs lack acousticness and instrumentalness. Tracks released in recent years tend to be more popular.

2. What is the overall trend and impact of notable events on the music industry?
<img width="661" alt="image" src="https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/2f68d302-ad59-419d-9eb6-d554553d609d">
This graph illustrates the growing popularity of music over time, highlighting the current generation’s increased inclination towards music and the positive impact of digitization on the music industry. Notably, the graph indicates a significant decline in 2001, coinciding with the detrimental effects of the terrorist attacks. Furthermore, the music industry has thrived during the COVID-19 pandemic, with people seeking solace in music, resulting in its continued growth.

3. Who are the most popular artists?

A. Word Cloud for the most popular artists
<img width="631" alt="image" src="https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/fd4bd31f-1f24-4b5a-9ca2-9177a8fa3066">

B. Bar Plot of the most popular artists
<img width="686" alt="image" src="https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/f5e8d668-1fec-49e3-9b38-a660ee7ed598">

4. What are the most popular songs in the dataset?
<img width="672" alt="image" src="https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/b1b1a72b-7f69-4be6-aa60-524473e35758">

This project aims to address the following key challenges:

1. **Feature Selection**: Identifying the most relevant audio features and metadata that contribute significantly to the popularity of songs on Spotify.

2. **Data Preprocessing**: Cleaning and preprocessing the dataset to handle missing values, outliers, and inconsistencies in order to create a reliable and representative training dataset.

3. **Model Selection and Training**: Exploring various machine learning algorithms and models such as regression, classification, or ensemble techniques to develop a predictive model that can accurately estimate the popularity of songs.

4. **Performance Evaluation**: Evaluating the performance of the developed model using appropriate metrics such as mean squared error, mean absolute error, or accuracy to assess its predictive capability and identifying areas of improvement.

5. **Interpretability**: Analyzing the model to gain insights into the key features and factors that influence song popularity, providing valuable information to artists, music labels, and Spotify itself for decision-making and enhancing the user experience.

## Resampling

We first split the target variable (‘highly_popular’) into X and y matrices. Then the RandomOverSampler module is used to select observations from the dataset at random as long as they have the same number of classes (Peixeiro, 2020). Therefore, we have 75,843 observations of class 0 and 75,843 observations of class 1. They are transformed into balanced data.
![image](https://github.com/Abhiashu10/Spotify-Popularity-Analysis-Prediction/assets/101308486/ed9bb8d1-c707-47aa-9270-b9e637a8b296)


## Pipeline building
Pipeline is an effective way to change and manipulate a lot of data. Pipelines are a series of ways to process data. With Pandas pipeline feature, we can string together different user-defined Python functions to make a pipeline for processing data.
Here we have created 3 small pipelines. One for ‘duration’, one for categorical variables and one for numerical variables.   
The first pipeline transforms the 'duration' feature. Checking and replacing the missing values of this characteristic with the median value is the initial step.   We use SimpleImputer for this purpose. Then, since this characteristic is initially recorded in milliseconds in the dataset. As a result, it has a vast range of possible values, up to 5 million. Therefore, we attempt to translate this characteristic to 'minute' by dividing each value by (1000 * 60). Finally, we scale this feature using sklearn's RobustScaler(). This scaler is extremely tolerant of outliers.   
The second pipeline consists of converting categorical variables to numerical variables using SimpleImputer for missing values and OneHotEncoder. The most frequent value is used to replace any missing data. OneHotEncoder is then utilized to encode them into spare matrices. However, because there are no categorical columns in the dataset that can be used with this pipeline, we are unable to use it at this time.      
The third pipeline is responsible for transforming numeric variables. First, missing data is replaced with the median value. Then, we utilize the RobustScaler() function to scale these attributes. Here, we employ all numerical characteristics because they are song qualities and, after analysis, they are suitable for this categorization problem.  

## Conclusion

As part of this project, we completed all necessary data science stages and were able to get some fascinating insights from the data. First, we performed the visualization and drew several intriguing insights from the data, such as the overall trend of the data, the events that had the greatest impact on the music business, the most well-known musicians, and the most well-liked songs. Three pipelines have been put in place to transform features in our models. Transforming duration from a millisecond to a minute is the first pipeline. The second pipeline uses a single hot encoder to convert category data to numerical variables, while the third pipeline scales the features of numerical variables. For this scenario, we employed 3 models (Logistic Regression, Random Forest and XGBoost) and Random Forest performed best with an accuracy of almost 97%. We discovered that the most crucial features to consider are loudness, energy, valence, acousticness, and explicitness. These features have the greatest influence on popularity. After hyperparameter tuning, we didn't see any appreciable improvements in the model's performance, which may be attributed to the large dataset's size, the lengthy training times associated with each combination, and the laptop's technical limitations. Working with this dataset was really intriguing, and in the future, we'd like to conduct our study utilizing cloud computing on a more powerful system. We'd also like to examine some of Spotify's rivals, such as Apple Music, Wynk Music, YouTube Music, etc. We also want to use Artificial Neural Networks, SVM, and K-nearest Neighbor for our analysis. For our study, we'd also like to make use of aspects like the music's genre, such as pop, rock, jazz, rap, etc.



