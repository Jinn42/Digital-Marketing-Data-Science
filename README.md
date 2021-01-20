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

## Chapter8

## Chapter9

## Chapter10

## Chapter11

## Chapter12
