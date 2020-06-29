"""Note this is taken for Jeremy class: udemy"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
class NBFeaturer(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)
    #fit the transformation
    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self
    # transform the
    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

# define the lemmatizer class
from nltk.stem import WordNetLemmatizer
class Lemmatizer(BaseEstimator):
    def __init__(self):
        self.l = WordNetLemmatizer()

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = map(lambda r: ' '.join([self.l.lemmatize(i.lower()) for i in r.split()]), x)
        x = np.array(list(x))
        return x