Data Science project - Bank Marketing Analysis
===

## Objective Setting
* When the bank conducts regular deposit marketing, it actually predicts whether the customer will apply for a regular deposit based on information such as the age of the customer and whether they will marry.

------------

## Data Curation
* We used the bank marketing dataset from [Kaggle](https://www.kaggle.com/janiobachmann/bank-marketing-dataset).

------------

## Data Inspection
> ### Head data
![1](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/1.PNG)

> ### Numerical column information
![2](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/2.png)

> ### Columns & data type
![3](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/3.png)

> ### Target value
![4](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/4.PNG)
* Approximately 5,200 customers have applied for regular deposits in the data set, accounting for 47% of the total.

> ### Categorical columns
![5](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/5.png)
![6](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/6.PNG)
* You can check the wrong data included in the columns.

> ### Numerical columns
![7](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/7.png)
* The distribution graphs in the Campaign, previcious, and pdays columns show that the outliers exist in that columns.

------------

## Data Preprocessing
* Dirty data was processed in three ways, Drop, fill, and replace, for missing, wrong, and outlier data.
> ### Wrong data
* Data unavailable due to case-sensitive -> Change uppercase to lowercase   
![8](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/8.PNG)      
* Replace wrong data to NA  
![9](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/9.PNG)     

------------
> ### Missing data
> #### Drop missing data
![10](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/10.PNG)
* If more than five missing values were found in one customer data, we decided that the data could not be used and dropped the row.

------------
> #### Replace missing value
1. After grouping by target value,      
**Missing data of the numerical feature is replaced by the median.**
![11](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/11.PNG)
* The numetrical column 'age' replaces the missing value using median value of the age column of the customers who select 'yes' in the deposit column and median value of the age column of the customers who select 'no' in the deposit column.    

**Missing data of the categorical feature is replaced by the mode.**
![12](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/12.PNG)
* The numetrical column 'marital' replaces the missing value using mode value of the marital column of the customers who select 'yes' in the deposit column and mode value of the marital column of the customers who select 'no' in the deposit column.

------------

2. After grouping by related feature,   
**Missing data of the numerical feature is replaced by linear regression predictions or median.**
* Median value
![13](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/13.PNG)
* Linear regression predictions
![14](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/14.PNG)
**Missing data of the categorical feature is replaced by the mode.**
![15](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/15.PNG)

------------

> ### Outliers
* pdays column
![16](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/16.PNG)
* The pdays column, which refers to the number of days elapsed from the date of contact, has a value of -1 of 74.5%. This column was dropped because the meaning of -1 could not be clearly identified.
* campaign and previous column
![17](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/17.PNG)
* Outlier data can also be found in the Campaign and pre-viable columns. However, because the column was considered important for predicting the results, data with values above 35 and above 40 were considered noise in each column and replaced with a median value.

------------

> ### Encoding
* The category column was encoded using LabelEncoder.

------------

> ### Scaling
* The numerical column was scaled using MinMaxScaler.

------------


## Data Analysis
* Use KNN, Decision tree, and XGBoost

------------

## Data Evaluation
> ### Defalut vs Tuned parameter
![18](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/18.PNG)
* You can check for higher accuracy when using a classifier that tunes the parameter.

------------
> ### Way of preprocessing
![19](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/19.PNG)
* When two pre-processed methods were applied, the second method showed a slightly higher acuity with the decision tree and xgboost models.

------------
> ### Comparison of three algorithms
![20](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/20.PNG)
* Using three models, XGBoost showed the highest prediction accuracy.
* When we used the bagging method, we could see that the decision tree was slightly higher while the knn was slightly lower.
* When the three models were applied to the majority voting method, they were able to check for lower accuracy than xgboost.

------------
> ### Confusion matrix
* KNN
![21](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/21.png)
* Decision tree
![22](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/22.png)
* XGBoost
![23](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/23.png)

------------

## Conclusion
![24](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/24.png)
* Using XGBoost, we looked at the impact of the columns on the prediction of the target value and found that contact, poutcom, and duration had the highest importance.
![25](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/25.png)
* The more successful the previous marketing campaign, the higher the rate of regular deposit applications.
![26](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/26.png)
* The more contacts were made before the campaign, the higher the rate of regular deposit applications.
![27](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/27.png)
* The longer the contact time, the higher the deposit application rate.
![28](https://github.com/JunHeon-Ch/DataScience_Bank_Marketing_Analysis/blob/master/wiki_image/28.png)
* However, we could see that the more contact attempts, the lower the deposit application rate.

------------

## Member Information & Role
||최준헌 | 양희림 | 김지현 |
|:-|:-:|:-:|:-:|
Student ID| 201533673 | 201735853 | 201633310 |
Email |chjh121@gmail.com|qkq1002@naver.com|zizi39028@gmail.com|
Role |Data preprocessing, Data Analysis, Data Evalution, Conclusion|Data preprocessing, Dataset management, Generate invalid data, Final ppt|Proposal ppt, Graph visualization, Data preprocessing, Final presentation|
