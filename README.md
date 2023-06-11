## 🚀 About Me
I'm Ashutosh pursuing Masters in applied machine intelligence and as part of my data maining course I have utlized machine learning techiniques for Spotify song Popularity Analysis & Prediction. We have also craeted a pipeline which is an effective way to change and manipulate a lot of data. Pipelines are a series of ways to process data. With Pandas pipeline feature, we can string together different user-defined Python functions to make a pipeline for processing data.
Here we have created 3 small pipelines.



## Spotify Track Popularity – Analysis and Prediction

The objective of this project is to develop a machine learning model that accurately predicts the popularity of songs on the Spotify platform. Spotify is a popular music streaming service with a vast library of songs from various genres and artists. The popularity of a song is a crucial factor for both artists and users, as it determines the visibility, reach, and potential success of a song. With about 158 million paying members in 178 countries, Spotify is the global leader in the streaming music sector. Spotify makes APIs available for obtaining data from its music track catalog. Spotify music datasets have become one of the most prominent datasets in the data science field for learning predictive modeling.

We have conducted EDA and applied supervised and unsupervised ML methods to find the most popular songs and also we string together different user-defined Python functions to make a pipeline for processing data, we have created 3 small pipelines. We used over 600,000 Spotify tracks with their attributes in our project. Each song has 20 columns, each of which describes a different component of the song. We are attempting to optimize several business challenges and expect that our inquiry will reveal some business-related insights. So, here are some business questions we would like to have answered:
1.	What is the overall trend and impact of notable events on the music industry?
2.	Who are the most popular artists in the dataset?
3.	What are the most popular songs in the dataset?
4.	What types of machine learning models do well on this dataset?
5.	What characteristics most clearly characterize a popular song?

This project aims to address the following key challenges:

1. **Feature Selection**: Identifying the most relevant audio features and metadata that contribute significantly to the popularity of songs on Spotify.

2. **Data Preprocessing**: Cleaning and preprocessing the dataset to handle missing values, outliers, and inconsistencies in order to create a reliable and representative training dataset.

3. **Model Selection and Training**: Exploring various machine learning algorithms and models such as regression, classification, or ensemble techniques to develop a predictive model that can accurately estimate the popularity of songs.

4. **Performance Evaluation**: Evaluating the performance of the developed model using appropriate metrics such as mean squared error, mean absolute error, or accuracy to assess its predictive capability and identifying areas of improvement.

5. **Interpretability**: Analyzing the model to gain insights into the key features and factors that influence song popularity, providing valuable information to artists, music labels, and Spotify itself for decision-making and enhancing the user experience.



## 🔗 Links
[![Code](spotify-track-popularity-analysis-and-prediction.ipynb)



## Features

This collection includes audio metadata for over 600,000 Spotify tracks. Each column focuses on a different facet of the song. The following is a description of them.

•	id: Spotify's unique identifier for each track (randomly generated alphanumeric string).    
•	popularity: The popularity of a song is measured on a normalized scale of [0-100], with 100 being the most popular.  
•	duration_ms: track duration in milliseconds.  
•	explicit: whether or not the song contains explicit content.   
•	artists: the artist's name.   
•	id_artists: Spotify's unique identification for each artist.   
•	release_date: When the album was released (date format: yyyy/mm/dd).    
•	danceability: indicates a track's suitability for dancing based on a variety of musical aspects such as tempo, rhythm stability, beat strength, and overall regularity.         A value of 0.0 indicates that it is the least danceable, while 1.0 indicates that it is the most danceable.    
•	energy: Energy is a perceptual measure of intensity and activity that ranges from 0.0 to 1.0. Typically, energetic tracks have a quick, loud, and boisterous vibe to them.   
•	key: The estimated overall key of the track. Integers are assigned to pitches using conventional Pitch Class notation. For example, 0 equals C, 1 equals C/D, 2 equals D, and so on. If no key was found, the value is set to -1.    
•	loudness: The overall volume of a track measured in decibels (dB). Typical values vary between -60 and 0 dB.   
•	mode: The mode of a recording (major=1 or minor=0) denotes the type of scale from which its melodic content is formed.    
•	speechiness: Speechiness is a value from 0.0 to 1.0 that shows how many spoken words are in a track. If a song's Speechiness is above 0.66, it probably has spoken words. If it is between 0.33 and 0.66, it could have both music and words. If it is below 0.33, it does not have any words.    
•	acousticness: a confidence scale ranging from 0.0 to 1.0 indicating whether or not the track is acoustic. 1.0 indicates a high degree of certainty that the track is acoustic.    
•	instrumentalness: The number of vocals in the song is represented by the instrumentalness, which ranges from 0.0 to 1.0. The closer to 1.0 it is, the more instrumental the music.     
•	liveness: The possibility of an audience being present in the recording ranges from 0.0 to 1.0. Higher liveness numbers indicate a greater likelihood that the track was performed live.      
•	valence: Valence is a scale from 0.0 to 1.0 that describes the melodic positivity expressed by a song.       
•	tempo: A track's overall approximate tempo in beats per minute (BPM).     
•	time_signature: A track's estimated total time signature. The time signature (meter) is a notational convention that specifies the number of beats in each bar (or measure).

## Resampling

We first split the target variable (‘highly_popular’) into X and y matrices. Then the RandomOverSampler module is used to select observations from the dataset at random as long as they have the same number of classes (Peixeiro, 2020). Therefore, we have 75,843 observations of class 0 and 75,843 observations of class 1. They are transformed into balanced data.

## Pipeline building
Pipeline is an effective way to change and manipulate a lot of data. Pipelines are a series of ways to process data. With Pandas pipeline feature, we can string together different user-defined Python functions to make a pipeline for processing data.
Here we have created 3 small pipelines. One for ‘duration’, one for categorical variables and one for numerical variables.   
The first pipeline transforms the 'duration' feature. Checking and replacing the missing values of this characteristic with the median value is the initial step.   We use SimpleImputer for this purpose. Then, since this characteristic is initially recorded in milliseconds in the dataset. As a result, it has a vast range of possible values, up to 5 million. Therefore, we attempt to translate this characteristic to 'minute' by dividing each value by (1000 * 60). Finally, we scale this feature using sklearn's RobustScaler(). This scaler is extremely tolerant of outliers.   
The second pipeline consists of converting categorical variables to numerical variables using SimpleImputer for missing values and OneHotEncoder. The most frequent value is used to replace any missing data. OneHotEncoder is then utilized to encode them into spare matrices. However, because there are no categorical columns in the dataset that can be used with this pipeline, we are unable to use it at this time.      
The third pipeline is responsible for transforming numeric variables. First, missing data is replaced with the median value. Then, we utilize the RobustScaler() function to scale these attributes. Here, we employ all numerical characteristics because they are song qualities and, after analysis, they are suitable for this categorization problem.  

## Conclusion

As part of this project, we completed all necessary data science stages and were able to get some fascinating insights from the data. First, we performed the visualization and drew several intriguing insights from the data, such as the overall trend of the data, the events that had the greatest impact on the music business, the most well-known musicians, and the most well-liked songs. Three pipelines have been put in place to transform features in our models. Transforming duration from a millisecond to a minute is the first pipeline. The second pipeline uses a single hot encoder to convert category data to numerical variables, while the third pipeline scales the features of numerical variables. For this scenario, we employed 3 models (Logistic Regression, Random Forest and XGBoost) and Random Forest performed best with an accuracy of almost 97%. We discovered that the most crucial features to consider are loudness, energy, valence, acousticness, and explicitness. These features have the greatest influence on popularity. After hyperparameter tuning, we didn't see any appreciable improvements in the model's performance, which may be attributed to the large dataset's size, the lengthy training times associated with each combination, and the laptop's technical limitations. Working with this dataset was really intriguing, and in the future, we'd like to conduct our study utilizing cloud computing on a more powerful system. We'd also like to examine some of Spotify's rivals, such as Apple Music, Wynk Music, YouTube Music, etc. We also want to use Artificial Neural Networks, SVM, and K-nearest Neighbor for our analysis. For our study, we'd also like to make use of aspects like the music's genre, such as pop, rock, jazz, rap, etc.




