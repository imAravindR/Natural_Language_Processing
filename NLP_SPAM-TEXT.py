# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:38:25 2018

@author: Aravind
"""

# Natural Language Processing
# Spam Text Message : https://www.kaggle.com/team-ai/spam-text-message-classification/data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('SPAM text message 20170820 - Data.csv')

# Check for missing values
data.isnull().sum()

# value counts 
data.Category.value_counts()

# Cleaning 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
corpus = []
for i in range(0, 5572):
    message = re.sub('[^a-zA-Z]', ' ', data['Message'][i])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus.append(message)
    
# Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
bag = CountVectorizer(max_features = 5000)
X = bag.fit_transform(corpus).toarray()
y = data.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# accuracy
crt_values = (y_pred == y_test).sum()
wrong_values = (y_pred != y_test).sum()
total = crt_values+wrong_values
result = crt_values/total
print(result) # 86% accuracy

