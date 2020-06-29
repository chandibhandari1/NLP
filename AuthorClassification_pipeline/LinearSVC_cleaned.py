"""
This is the Authoer Classiciation with linear SVC and CountVectorizer after cleaning
gives 74% accuracy worser than before cleaned: cleaning is removing the imp infos
Need to modified the cleaner
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



# import clean_text function from teh severFunction module
import severalFunction as funPool
cleaner = funPool.severalFunction()

# Now reading the data
train_path = 'train/train.csv'
df = pd.read_csv(train_path)
df['cleanText'] = df['text'].apply(cleaner.clean_text)

# Again import the test data
test_path = 'test/test.csv'
test_data = pd.read_csv(test_path)
# apply cleaner
test_data['cleanText']=test_data['text'].apply(cleaner.clean_text)
print(test_data.head())

# train-test split
target = 'author'
X=df['cleanText']
y=df[target]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=3)
# check the size
print(X.shape, X_train.shape, X_test.shape)
print(' ')
# check the X_train head
print(X_train.head())

# build the feature matrices
ngram_counter = CountVectorizer(ngram_range=(1, 4), analyzer='char')
X_train_cv = ngram_counter.fit_transform(X_train)
X_test_cv  = ngram_counter.transform(X_test)

# train the classifier
classifier = LinearSVC()
model = classifier.fit(X_train_cv, y_train)
# test the classifier
pred_y = model.predict(X_test_cv)
# get the accuracy score
print(accuracy_score(pred_y, y_test)*100)


# prediction for the test_data
test_data_cv = ngram_counter.transform(test_data['cleanText'])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%% get the actual labels for the given docs
#Prediction on validation set
pred_val = model.predict(test_data_cv)
#%%
#Create a  DataFrame with the ids and our prediction regarding whether they disenrolled or not
IDwitPrediction_clf=pd.DataFrame({'Id':test_data['id'], 'Pred2':pred_val})
print(IDwitPrediction_clf.head())