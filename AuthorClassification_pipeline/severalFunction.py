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