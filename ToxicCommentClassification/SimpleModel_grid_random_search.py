import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

# read the train data
data_path = 'train/train.csv'
df = pd.read_csv(data_path)

# extending the display option
pd.set_option('display.width', 200)
np.set_printoptions(linewidth=2)
pd.set_option('display.max_columns',10)

# For simplicity select 10000 data points only
df =df[:5000]
print(df.shape)
print(df.head())
print(df.toxic.value_counts())
# Select predictor and target
X = df['comment_text'].values
y = df['toxic'].values
# importing the lemmatizer and baseEstimator
import myModule as mmd
NBsel = mmd.NBFeaturer(1)
lemT = mmd.Lemmatizer()

# call the model and TFIDFvectorizer
tfidf = TfidfVectorizer(max_features=2000)
logReg = LogisticRegression()

# first pipe with only tfidf and LogReg: (name, actual_fun)
pipe1 = Pipeline([
    ('mytfidf', tfidf),
    ('LogReg', logReg)
])

# now do the cross validation with pipe1
print(cross_val_score(estimator=pipe1, X=X, y=y, scoring='roc_auc', cv=3))

print("***********The pipe:2 cross validation started**********")
# second pipe with lemmatizer, tfidf,NBfeature, and logistic regression
pipe2 = Pipeline([
    ('lemmatize',lemT),
    ('mytfidf', tfidf),
    ('NBfeature', NBsel),
    ('LogReg', logReg)
])

# now do the cross validation with pipe1
print(cross_val_score(estimator=pipe2, X=X, y=y, scoring='roc_auc', cv=3))


print("***********The pipe:3 cross validation started**********")
"""
create the additional tfidf vectorizer for chars and join with word vectorizer using FeatureUnion
"""
# create the feature combine and then create the pipe
tfidf_word = TfidfVectorizer(max_features=2000, analyzer='word')
tfidf_char = TfidfVectorizer(max_features=2000, analyzer='char')
# feature union
feat_comb = FeatureUnion([
    ('tfidf_word', tfidf_word),
    ('tfidf_char', tfidf_char)
])
# create the third pipe with including the feature union
pipe3 = Pipeline([
    ('lemmatize',lemT),
    ('combFeature', feat_comb),
    ('NBfeature', NBsel),
    ('LogReg', logReg)
])

# now do the cross validation with pipe1
print(cross_val_score(estimator=pipe3, X=X, y=y, scoring='roc_auc', cv=3))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Random and Grid search for parameter%%%%%%%%%
# first get the key
print(pipe3.get_params().keys())

print('****************Grid search for optimal paramters *************')

param_grid = [{
    'combFeature__tfidf_word__max_features': [2000, 3000],
    'combFeature__tfidf_char__stop_words': [2000, 3500],
    'LogReg__C': [2.],
}]

grid = GridSearchCV(pipe3, cv=3, param_grid=param_grid, scoring='roc_auc',n_jobs=1,
                            return_train_score=False, verbose=1)
grid.fit(X, y)
print(grid.cv_results_)
# lets get the best parameter
print(grid.best_params_)
# print the best score
print(grid.best_score_)


print('**************** Random search for optimal paramters *************')

param_grid = [{
    'combFeature__tfidf_word__max_features': [2000, 3000, 5000],
    'combFeature__tfidf_char__stop_words': [2000, 3500, 5000],
    'LogReg__C': [1., 2., 5.],
}]

grid = RandomizedSearchCV(pipe3, cv=3, n_jobs=1, param_distributions=param_grid[0], n_iter=1,
                          scoring='roc_auc', return_train_score=False, verbose=1)
grid.fit(X, y)
print(grid.cv_results_)
# lets get the best parameter
print(grid.best_params_)
# print the best score
print(grid.best_score_)