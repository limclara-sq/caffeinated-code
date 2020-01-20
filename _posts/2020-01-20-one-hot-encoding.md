---
layout: post
title: "Encoding categorical attributes correctly in your ML project "
author: "Clara Lim"
categories: journal
#tags: [documentation,sample]
#image: project-infinity.jpg
---

One of the most common tasks in the Feature Engineering stage of your Machine Learning project is handling categorical attributes. Many ML models, such as regression or support vector machines, are algebraic; they require a numerical input. Additionally, not all ML packages automatically transform categorical data to numerical data. As such, to work with these categorical attributes, we need to transform them into numeric labels, then apply some form of encoding scheme on them.

### Understanding categorical data representations

Categorical features typically take on a fixed number of possible values. The major types of categorical variables are (i) nominal variables and (ii) ordinal variables.

Nominal variables have no intrinsic ordering within their categories. For example, a dataset on travel destinations can have countries as a categorical feature. There should not be any mathematical order of the labels within this "country" feature (e.g. Singapore vs United Kingdom vs China). On the other hand, ordinal variables have some order associated with them naturally. For a temperature feature with labels 'high', 'medium' and 'low', it makes sense that a 'low' temperature is mathematically smaller than a 'high' temperature.

It is important to know whether a variable is ordinal or nominal because statistical analyses usually assume that variables have specific levels of measurement. In addition, when dealing with encoding of categorical variables, we must be careful not to introduce order to features which have no natural ordering.

### The grave danger of introducing order to your nominal categorical variables

As we mentioned, categorical labels need to be encoded to numerical labels so that the ML models are able to understand them. However, when the categorical feature is nominal, one would be careful not to blindly label encode these features. As an illustration, let's go back to the example of the  3 labels ("Singapore", "United Kingdom" and "China") in a dataset. Label encoding would mean that each value is converted to a number, such as:

`{"Singapore": 1, "United Kingdom": 2, "China": 3}`

While we understand that no such ordering of these labels really exists, our ML model might not intuitively understand that. If we were to feed these numerical labels directly into a model, the cost function is likely to be affected by these values.

### One-Hot Encoding!

One-Hot Encoding can help us model this understanding of ours mathematically. What is One-Hot Encoding? Looking on Scikit-learn's documentation, we see:

> Encode categorical features as a one-hot numeric array.

and

> The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)

In other words, each label within the categorical feature is transformed into a binary column, and we end up with `n` new columns representing the original column of `n` different labels.

| country      | GDP per capita (USD) |
| ----------- | ----------- |
| Singapore      | 57 714       |
| United Kingdom   | 39 720        |
| China   | 8826        |

| GDP per capita (USD) | country_singapore | country_uk | country_china |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 57 714       | 1 | 0 | 0 |
| 39 720       | 0 | 1 | 0
| 8826         | 0 | 0 | 1 |

The benefit of this method is in preventing weighting a value improperly.


### A Scikit-Learn Implementation
