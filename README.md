# Digital-Marketing-Data-Science
## This repository is concerned about the application of data science son Digital marketing, inspired by the book 'Hands-On Data Science for Marketing'(Author:Yoon Hyup Hwang). I did the projects in python and I modifed some codes. Here are some introduction and personal notes.

3 types of analysis: Descriptive versus explanatory versus predictive analyses


## Chapter1
content: basic operations

packages: numpy;LogisticRegression from sklearn.linear_model; matplotlib.pyplot 

## Chapter2
content: KPI tracking - converison rate 
Conversion rates by Number of contact& Age group & Marital Status
pie chart, bar chart, box chart, line chart

packages: pandas; matplotlib.pyplot 

## Chapter3
content: Drivers behind Marketing Engagement
pivot table for engagement for offer type/ sales channel.
Regression Analysis with Both Continuous and Categorical Variables(sm.logit)
Three ways to handle categorical variables: 
1. factorize (labels, levels = df['Education'].factorize())
2. pandas' Categorical variable series (categories = pd.Categorical(df['Education'], categories=['High School or Below', 'Bachelor', 'College']))
3. dummy variables (pd.get_dummies(df['Education']))

packages: pandas; matplotlib.pyplot; sm from statsmodels.formula.api;

## Chapter4
content: From Engagement to Conversion - decision tree model
This chapter discusses how to use different machine learning models to understand what drives conversion.

packages:pandas; matplotlib.pyplot; from sklearn import tree;import graphviz;from IPython.core.display import display, HTML


## Chapter5
content: Exploratory Product Analytics
various data aggregation and analysis methods in Python to obtain further insights into the trends and patterns in products. 
Time series numeber of orders/ repeat customers; popular items over time;

packages: pandas; matplotlib.pyplot;

## Chapter6
content: Recommending the Right Products
How to improve product visibility and recommend the right products that individual customers are most likely to purchase. How to use the collaborative filtering algorithm in Python in order to build a recommendation model. Then, it covers how these recommendations can be used for marketing campaigns.
use pivot table to create customer-item matrix, and then 
1. User-based Collaborative Filtering
user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))

2. Item-based Collaborative Filtering
item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))

packages: pandas; from sklearn.metrics.pairwise import cosine_similarity

## Chapter7
content: Exploratory Analysis for Customer Behavior
various metrics that can be used to analyze how customers behave and interact with the product.
Engagement Rates by Offer Type;Engagement Rates by Sales Channel;Engagement Rates by Months Since Policy Inception

packages:import matplotlib.pyplot;pandas 

## Chapter8
content: Predicting the Likelihood of Marketing Engagement
Model:RandomForestClassifier; Evaluating Metrics: Accuracy, Precision, and Recall; ROC & AUC
accuracy_score(y_train, in_sample_preds);accuracy_score(y_test, out_sample_preds);precision_score(y_train, in_sample_preds));precision_score(y_test, out_sample_preds));recall_score(y_train, in_sample_preds));recall_score(y_test, out_sample_preds))
in_sample_fpr, in_sample_tpr, in_sample_thresholds = roc_curve(y_train, in_sample_preds)
out_sample_fpr, out_sample_tpr, out_sample_thresholds = roc_curve(y_test, out_sample_preds)
in_sample_roc_auc = auc(in_sample_fpr, in_sample_tpr)
out_sample_roc_auc = auc(out_sample_fpr, out_sample_tpr)

packages:import pandas as pd;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split; from sklearn.ensemble import RandomForestClassifier;from sklearn.metrics import accuracy_score, precision_score, recall_score;from sklearn.metrics import roc_curve, auc

## Chapter9
content: Customer Lifetime Value
use linear regression(or svm, randomforestregressor) model to predict CLV and evaluate the model with r2_score and median_absolute_error.

packages: 
from sklearn.linear_model import LinearRegression;from sklearn.svm import SVR;from sklearn.ensemble import RandomForestRegressor; from sklearn.metrics import r2_score, median_absolute_error

## Chapter10
content: Customer Segmentation
clustering algorithms (k-means) to build different customer segments and use silhouette_score to select best number of clusters.

packages:from sklearn.cluster import KMeans,from sklearn.metrics import silhouette_score

## Chapter11
content: Customer Retention
use ANN to build model to predict if the customer will churn or not. and then use 5 classic metrics to verify the result

model = Sequential()
model.add(Dense(16, input_dim=len(features), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

packages:from keras.models import Sequential;from keras.layers import Dense;from sklearn.model_selection import train_test_split;from sklearn.metrics import accuracy_score, precision_score, recall_score; from sklearn.metrics import roc_curve, auc

## Chapter12
content: ABtesting
AB tesing on 5 variables:marketsize, location, age of store, promotion and week, and then analyse the distribution and relation of variables
Statistical Significance: to test the if there is significant impact of different variabels on results
t, p = stats.ttest_ind(
    df.loc[df['Promotion'] == 1, 'SalesInThousands'].values, 
    df.loc[df['Promotion'] == 2, 'SalesInThousands'].values, 
    equal_var=False
)

packages: import matplotlib.pyplot as plt;import pandas as pd;import numpy as np;from scipy import stats

