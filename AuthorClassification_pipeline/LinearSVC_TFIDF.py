""" When Running others methond Linear SVC found to be working best so far"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import pickle

# Warning ignore
import warnings
warnings.filterwarnings('ignore')

import sys

# Read the file
train_path = 'train/train.csv'
df = pd.read_csv(train_path)
# due to memory issue with large numpy array, I just selected 5000 data
df = df[:5000]
#df = shuffle(df)
print(df.head())
print(df.shape)
print(df.info())




print('\nPrinting the value:\n')
print(df.author.value_counts(dropna=False))

df['Category_label'] = df['author'].factorize()[0]
from io import StringIO
Category_label_df = df[['author', 'Category_label']].drop_duplicates().sort_values('Category_label')
Category_to_label = dict(Category_label_df.values)
label_to_Category = dict(Category_label_df[['Category_label', 'author']].values)
print('\n New data: \n',df.head())

print(Category_to_label)
print(label_to_Category)
print(Category_label_df.head())

# import scikit-learn packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# now create the tfidf
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                        ngram_range=(1, 2), stop_words='english')

# fit the tfidf to the data
features = tfidf.fit_transform(df.text).toarray()
labels = df.Category_label
print(features.shape)
print(labels.unique())

# Train test split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.2, random_state=2)

# model fitting
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print the report
print(classification_report(y_test, y_pred, target_names=df['author'].unique()))


# confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

sys.exit(0)
# plotting the heatmap for confusion matrix
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_mat,
            annot=True,
            xticklabels=df['author'].values,
            yticklabels=df['author'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()

# dumping TFIDF and model
pickle.dump(tfidf, open("tfidf.pickle", "wb"))
pickle.dump(model, open("model.pickle", "wb"))