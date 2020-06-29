"""
This is the Authoer Classiciation with linear SVC and CountVectorizer with pipeline
gives 78% better
"""
# importing all the required packages
import numpy as np
import pandas as pd
import re
import sys
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, maxabs_scale, MinMaxScaler, power_transform, quantile_transform
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score


# Now reading the data
train_path = 'train/train.csv'
df = pd.read_csv(train_path)


# Again import the test data
test_path = 'test/test.csv'
test_data = pd.read_csv(test_path)
print(test_data.head())

# train-test split
target = 'author'
X=df['text']
y=df[target]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=3)
# check the size
print(X.shape, X_train.shape, X_test.shape)
print(' ')
# check the X_train head
print(X_train.head())


# now create the pipline: remember first count vectorizer and model (name, action)
# import pipeline
from sklearn.pipeline import Pipeline
pipe_ch = Pipeline([
              ('ngram', CountVectorizer(ngram_range=(1, 4), analyzer='char')),
              ('clf',   LinearSVC())
      ])

# train the classifier: directly aply it to the X_train data
model = pipe_ch.fit(X_train, y_train)

#  Directly predic X_test the classifier
pred_y= model.predict(X_test)


# get the accuracy score
print(np.mean(pred_y==y_test))

# Now do the grid search for best parameters
from sklearn.model_selection import GridSearchCV

pg = {'clf__C': [0.1, 1, 10, 100]}

grid = GridSearchCV(pipe_ch, param_grid=pg, cv=5)
grid.fit(X_train, y_train)

# lets get the best parameter
print(grid.best_params_)
# print the best score
print(grid.best_score_)