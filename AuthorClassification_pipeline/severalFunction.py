"""
This files contains a several functions to utilized in AuthorClassify_ModelwithPipeline

"""
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
# create the self-defined stopwords by adding the your own words
own_stopwords =set(stopwords.words('english'))

class severalFunction:
    def __init__(self):
        pass

    # define the saveral functions which encapsulates different actions
    # Step 1: define function encapsulating the pre-processing steps:
    """This fu"""

    def data_preparation(self, data, text_col_name):
        """
        This function takes data with the column to process is: text_col_name, then removes punctuation, lower the text,
        count the word_length of cleaned_text, length (all characters) , words without stopwords, count commas, and avg word lenght
        """
        data['cleaned'] = data[text_col_name].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        # numerical feature engineering
        # no of commas in text
        data['commas'] = data[text_col_name].apply(lambda x: x.count(','))
        # count all character
        data['text_len'] = data['cleaned'].apply(lambda x: len(x))
        # count the words
        data['words'] = data['cleaned'].apply(lambda x: len(x.split(' ')))
        # count words without stopwords
        data['wordNoStopWords'] = data['cleaned'].apply(lambda x: len([words for words in x.split(' ')
                                                                       if words not in own_stopwords]))
        # avg word length-after removing stopwords
        data['avgWordLength'] = data['cleaned'].apply(
            lambda x: np.mean([len(words) for words in x.split(' ') if words not in own_stopwords]) if len(
                [len(words) for words in x.split(' ') if words not in own_stopwords]) > 0 else 0)

        return data


    # function to clean text

    # adding extra stop words
    new_stop_words1 = ['hello', 'hi', 'Harry']

    STOPWORDS = own_stopwords.union(new_stop_words1)

    # # Adding all the lists of States
    # new_stop_words2 = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'gu', 'hi', 'ia', 'id', 'il',
    #                    'in', 'ks','ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh',
    #                    'nj', 'nm','nv', 'ny', 'oh','ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vi',
    #                    'vt', 'wa', 'wi', 'wv', 'wy']
    #
    # STOPWORDS = STOPWORDS.union(new_stop_words2)

    def clean_text(self, text):
        # address and zipcode
        regexp1 = r"[0-9]{1,5} .+, .+, [A-Z]{2} [0-9]{5}"
        regexp2 = r"[0-9]{1,5} .+, .+, [A-Z]{2}"
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        REPLACE_NUM_BY_SPACE = re.compile('[0-9]')
        # the funciton begins here
        text = re.sub(regexp1, ' ', text)
        text = re.sub(regexp2, ' ', text)
        text = text.lower()
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub(' ', text)
        text = REPLACE_NUM_BY_SPACE.sub(' ', text)
        text = text.replace('\d+', ' ')
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
        return text

    #Note: df['Text'] = df['Text'].apply(clean_text)