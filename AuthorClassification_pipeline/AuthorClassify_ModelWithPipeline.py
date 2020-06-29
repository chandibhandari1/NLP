"""
This tutorial gives how to create the pipe line: with Scikit-learn pipeline
"""
# importing all the required packages
import numpy as np
import pandas as pd
import re
import sys
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, maxabs_scale, MinMaxScaler, power_transform, quantile_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

# import pipeline
from sklearn.pipeline import Pipeline

# Now reading the data
train_path = 'train/train.csv'
df = pd.read_csv(train_path)
# looking at the data
pd.set_option('display.width', 200)
#pd.set_option('display.max_colwidth', -1)
np.set_printoptions(linewidth=2)
pd.set_option('display.max_columns',10)
print(df.head())

# Now lets pull the data_preparation function from module severalFunction
# Check the text processing function to the data: data= df, text_col_name=text
import severalFunction as funPull
prep = funPull.severalFunction()
prep.data_preparation(data=df, text_col_name='text')
print(' ')

# get all the column names
target ='author'
col_names = df.columns.values
# since id is not require, text is cleaned, author is lebel
drop_columns = ['id','text','author']
features = [cols for cols in col_names if cols not in drop_columns]
numerical_features= [cols for cols in col_names if cols not in ['id','text','author','cleaned']]

# train-test split
X=df[features]
y=df[target]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=3)
# check the size
print(X.shape, X_train.shape, X_test.shape)
print(' ')
# check the X_train head
print(X_train.head())

# print the value category
print("Different author counts: \n", df['author'].value_counts())

# call the module that I defined for column selector from module columnSelector
import columnSelector as colSel

# %%%%%%%%%%%%%%%%%% Pipeline creation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
- To create the pipeline simply pass array of tuple: (name,object):name=name of action, object = exact action_to_perform
- for example in the followig mini pipe: 
      - i will take cleaned column and apply tfidf transformatio
      - two action: one - for selection of column, second- for applying transrom
      - I will provide name: ('selector', TextSelector) and ('tfidf', TfidfVectorizer) in pipe
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# lets utilized our textcolumn selector from my moduel columnSelector pipe with TFIDF
# mini-pipe (tfidf with textcolumn selector
clean_to_tfidf = Pipeline([
                ('selector', colSel.TextSelector(key='cleaned')),
                ('tfidf', TfidfVectorizer( stop_words='english'))
            ])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
method of apply: clean_to_tfidf.fit(X_train) to fit on training data
                 clean_to_tfidf.transform(X_train) to apply on training data
                 clean_to_tfidf.fit_transform(X_train) to do both at once
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%

clean_to_tfidf.fit_transform(X_train)

#Another mini pipiline:
"""
since all my numberical columns are heterogeneous we need to perform different action so create
different mini pipeline as follows
Note: since the transformer pipeline creates the matrix for the called column
"""

# mini-pipe 2: select text_len column and apply standardScalar
from sklearn.preprocessing import StandardScaler

text_len_scaled =  Pipeline([
                ('selector', colSel.NumberSelector(key='text_len')),
                ('standard', StandardScaler())
            ])
# fit_tranform the standardScalear on text_len column
text_len_scaled.fit_transform(X_train)

# Similarly create other mini-pipeline: create the pipe line for every features (variable)
words_scaled =  Pipeline([
                ('selector', colSel.NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
wordNoStopWords_scaled =  Pipeline([
                ('selector', colSel.NumberSelector(key='wordNoStopWords')),
                ('standard', StandardScaler())
            ])
avgWordLength_scaled =  Pipeline([
                ('selector', colSel.NumberSelector(key='avgWordLength')),
                ('standard', MinMaxScaler())
            ])
commas_scaled =  Pipeline([
                ('selector', colSel.NumberSelector(key='commas')),
                ('standard', StandardScaler()),
            ])
# Now we need to join them all together
# packages to combine features: now make a pipeline for all pipelines: call FeatureUnion processing for pipeline
# same structure: (name, actual_task)
from sklearn.pipeline import FeatureUnion
#combine and then apply all those transformations at once with a single fit, transform, or fit_transform call.
feat_comb = FeatureUnion([('clean_to_tfidf', clean_to_tfidf),
                      ('text_len_scaled', text_len_scaled),
                      ('words_scaled', words_scaled),
                      ('wordNoStopWords_scaled', wordNoStopWords_scaled),
                      ('avgWordLength_scaled', avgWordLength_scaled),
                      ('commas_scaled', commas_scaled)])

# now create the pipeline after combine the features at once
feature_processing = Pipeline([('feat_comb', feat_comb)])
feature_processing.fit_transform(X_train)

# Now adding the model into the full processed pipeline at the end: the syntax as same (see below)
# adding the classifier at the end:It should be end (order matter, clean then model fit)
final_pipeline = Pipeline([
    ('features',feat_comb),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

# Now fit the pipeline (with model) to the X_train and Y_train as follow
final_pipeline.fit(X_train, y_train)
# Now predict on the X_test with final_pipeline
pred_y = final_pipeline.predict(X_test)

# mean accuracy
print('Predicted Accuracy: \n',np.mean(pred_y == y_test))

# now call the accuracy score and confusion matrix
print(confusion_matrix(pred_y, y_test))

#%%%%%%%%%%%%%%%%%% Cross Validation on pipeline %%%%%%%%%%%%%%%%%%%%%%%%%
"""
To figure out: 
-no_of trees, how_deep_tree, how_many_words in tfidf, include stop_words or not in the model
- figuring out what the best hyperparameters of the data set is 
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#To see the list of all the possible things that could fine tune, call get_params().keys() on  pipeline.
print(final_pipeline.get_params().keys())

"""
define the dictionary what to change: get the key from pipeline as above
call the GridSearchCV 
"""
# get the key and provide the vlaue: for now keep sort and small num of keys and values
# have taken only 4 keys and 2 values for each key
hyperparameters = { 'features__clean_to_tfidf__tfidf__max_df': [0.9, 0.95],
                    'features__clean_to_tfidf__tfidf__ngram_range': [(1,1), (1,2)],
                   'classifier__max_depth': [50, 70],
                    'classifier__min_samples_leaf': [1,2]
                  }
# now apply the grid search 5 cross-validation samples
clf = GridSearchCV(final_pipeline, hyperparameters, cv=5)
# Fit and tune model
clf.fit(X_train, y_train)
# now check the best parameter
print(clf.best_params_)

#%%%%%%%%%%%%%%%%%%%%% Refitting the best hyperparameter on training data
"""
Just call refit to automatically fit the pipeline on all of the training data with the best_params_setting applied
Then applying it to the test data is the same as before.
"""

#refitting on entire training data using best settings
clf.refit

# now predict with model with best parameters
pred_y = clf.predict(X_test)
probs = clf.predict_proba(X_test)
# check the score
print(np.mean(pred_y == y_test))


#%%%%%%%%%%%%%%%%%%%%%%% Finally predict on the test data with all processing applied on training data
test_path = 'test/test.csv'
test_data = pd.read_csv(test_path)
cleaned_test_data = prep.data_preparation(data=test_data, text_col_name='text')
# now do the easy job: predict with pipeline
predictions = clf.predict_proba(cleaned_test_data)
# to get the probability of exactly one category: for the second category
#pred_prob = predictions[:,1]

# create the data frame with column name with target lebels and values with predicted probability
preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)
# if the best parameter was not there
#preds = pd.DataFrame(data = predictions, columns=clf.classes_)


# now concatenate with id
out_put = pd.concat([cleaned_test_data[['id']], preds], axis=1)
out_put.set_index('id', inplace = True)
print(out_put.head())

# with the predicted lebel in dataframe
# Note: this is just for understanding how we can get lebels by two methods
# Now get the predicted labels
pred_label_index = np.argmax(predictions,axis=1)
pred_lebel = clf.best_estimator_.named_steps['classifier'].classes_[pred_label_index]
#Prediction on validation set
pred_val = clf.predict(cleaned_test_data)
#%%
#Create a  DataFrame with the ids and our prediction regarding whether they disenrolled or not
IDwitPrediction_clf=pd.DataFrame({'Id':cleaned_test_data['id'],'Prediction':pred_lebel, 'Pred2':pred_val})

print(IDwitPrediction_clf.head())




