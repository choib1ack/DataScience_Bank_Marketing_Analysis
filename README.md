Driver License Simulation(운전 뽀개기)
===

## Objective Setting
* When the bank conducts regular deposit marketing, it actually predicts whether the customer will apply for a regular deposit based on information such as the age of the customer and whether they will marry.

## Data Curation
* We used the bank marketing dataset from kaggle.
![Kaggle](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)

## Data Inspection
> ### Head data
![1](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/1.PNG)
> ### Numerical column information
이미지2
> ### Columns & data type
이미지3
> ### Target value
이미지4
* Approximately 5,200 customers have applied for regular deposits in the data set, accounting for 47% of the total.
> ### Categorical columns
이미지5
이미지6
* You can check the wrong data included in the columns.
> ### Numerical columns
이미지7
* The distribution graphs in the Campaign, previcious, and pdays columns show that the outliers exist in that columns.

## Data Preprocessing
* Dirty data was processed in three ways, Drop, fill, and replace, for missing, wrong, and outlier data.
> ### Wrong data
이미지8
* Data unavailable due to case-sensitive -> Change uppercase to lowercase
이미지9
* Replace wrong data to NA

> ### Missing data
> #### Drop missing data
이미지10
* If more than five missing values were found in one customer data, we decided that the data could not be used and dropped the row.
> #### Replace missing value
1. After grouping by target value,
Missing data of the numerical feature is replaced by the median.
이미지11
* The numetrical column 'age' replaces the missing value using median value of the age column of the customers who select 'yes' in the deposit column and median value of the age column of the customers who select 'no' in the deposit column.

Missing data of the categorical feature is replaced by the mode.
이미지12
* The numetrical column 'marital' replaces the missing value using mode value of the marital column of the customers who select 'yes' in the deposit column and mode value of the marital column of the customers who select 'no' in the deposit column.

2. After grouping by related feature,
Missing data of the numerical feature is replaced by linear regression predictions or median.
* Median value
이미지13
* Linear regression predictions
이미지14
Missing data of the categorical feature is replaced by the mode.
이미지15

> ### Outliers
* pdays column
이미지16
* The pdays column, which refers to the number of days elapsed from the date of contact, has a value of -1 of 74.5%. This column was dropped because the meaning of -1 could not be clearly identified.
* campaign and previous column
이미지17
* Outlier data can also be found in the Campaign and pre-viable columns. However, because the column was considered important for predicting the results, data with values above 35 and above 40 were considered noise in each column and replaced with a median value.

> ### Encoding
* The category column was encoded using LabelEncoder.

> ### Scaling
* The numerical column was scaled using MinMaxScaler.


## Data Analysis
* Use KNN, Decision tree, and XGBoost

## Data Evaluation
> ### Defalut vs Tuned parameter
이미지18
* You can check for higher accuracy when using a classifier that tunes the parameter.
> ### Way of preprocessing
이미지19
* When two pre-processed methods were applied, the second method showed a slightly higher acuity with the decision tree and xgboost models.
> ### Comparison of three algorithms
이미지20
* Using three models, XGBoost showed the highest prediction accuracy.
* When we used the bagging method, we could see that the decision tree was slightly higher while the knn was slightly lower.
* When the three models were applied to the majority voting method, they were able to check for lower accuracy than xgboost.
> ### Confusion matrix
* KNN
이미지21
* Decision tree
이미지22
* XGBoost
이미지23

## Conclusion
이미지24
* Using XGBoost, we looked at the impact of the columns on the prediction of the target value and found that contact, poutcom, and duration had the highest importance.
이미지25
* The more successful the previous marketing campaign, the higher the rate of regular deposit applications.
이미지26
* The more contacts were made before the campaign, the higher the rate of regular deposit applications.
이미지27
* The longer the contact time, the higher the deposit application rate.
이미지28
* However, we could see that the more contact attempts, the lower the deposit application rate.

## Member Information & Role
||최준헌 | 양희림 | 김지현 |
|:-|:-:|:-:|:-:|
Student ID| 201533673 | 201735853 | 201633310 |
Email |chjh121@gmail.com|qkq1002@naver.com|zizi39028@gmail.com|
Role |Data preprocessing, Data Analysis, Data Evalution, Conclusion|Data preprocessing, Dataset management, Generate invalid data, Final ppt|Proposal ppt, Graph visualization, Data preprocessing, Final presentation|
