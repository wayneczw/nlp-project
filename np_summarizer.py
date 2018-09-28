from argparse import ArgumentParser, FileType
import argparse
from collections import OrderedDict
from IPython.display import display
import logging
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, TreebankWordTokenizer
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 10)
import random
import regex
import string
import os
import datetime
from collections import Counter
from utils import load_instances, load_dictionary_from_file, _process_regex_dict

# library for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.style.use('seaborn-whitegrid')
params = {'figure.figsize': (20,15),
            'savefig.facecolor': 'white'}
plt.rcParams.update(params)

logger = logging.getLogger(__name__)
EMOTICONS_REGEX = _process_regex_dict(load_dictionary_from_file('./emoticons.yaml'), regex_escape=True)
EMOTICONS_TOKEN = _process_regex_dict(load_dictionary_from_file('./emoticons.yaml'))
STOPWORDS = set(stopwords.words('english') + ["'s", "one", "use", "would", "get", "also"]) - {'not', 'no', 'won', 'more', 'above', 'very', 'against', 'again'}
SPECIAL_TOKEN = {"n't": 'not'}
IMAGES_DIRECTORY = './images'


data_path = './data/CellPhoneReview.json'
data_path = './data/sample_data.json'
# load data
import json
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['unixReviewTime', 'reviewTime'])


df['segmentedSentences'] = df['reviewText'].apply(seg_sentences)
df['sentenceCount'] = df['segmentedSentences'].apply(len)
original_sentences = flatten([[zipped[1]] * zipped[0] for zipped in zip(df['sentenceCount'], df['reviewText'])])
sentences_df = pd.DataFrame(original_sentences, columns = ['originalSentences']).reset_index().drop(columns = ['index'])
sentences_df['sentence'] = pd.Series(flatten(df['segmentedSentences']))
sentences_df['tokenizedSentence'] = sentences_df['sentence'].apply(tokenize, unique=False, freq=False, stopwords = False, remove_punc=False, lower=False)
sentences_df['posTagged'] = sentences_df['tokenizedSentence'].apply(pos_tag)

sentences_df.groupby('originalSentences').head()


def tokenize(sentence, lower=True, remove_punc=True, stopwords=True, keep_emo=True, stem = False, **kwargs):

    tokens = list()

    tokenized = TreebankWordTokenizer().tokenize(sentence)
    tokenized = [SPECIAL_TOKEN[token] if SPECIAL_TOKEN.get(token, '') else token for token in tokenized]

    if lower:
        tokenized = [token.lower() for token in tokenized]

    if stopwords:
        tokenized = [token for token in tokenized if token not in STOPWORDS]

    if keep_emo:
        tokens += _emoticons_detection(tokenized)
    else:
        tokens += tokenized

    if remove_punc:
        tokens = [token for token in tokens if token not in string.punctuation]

    if stem:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens
#end def

def _verify_emoticon(tmp_token, token):
    return (tmp_token + token) in EMOTICONS_TOKEN
#end def

def _emoticons_detection(tokenized_list, flag=False):

    new_tokenized_list = list()
    n = len(tokenized_list)
    i = 0
    emoticons_detected = False
    while i < n:
        token = tokenized_list[i]
        if len(token) != 1:
            new_tokenized_list.append(token)
        else:
            tmp_token = token
            for k in range(i+1, n):
                i += 1
                token = tokenized_list[k]
                if (len(token) == 1) & (k != (n - 1)):
                    tmp_token += token
                    if _verify_emoticon(tmp_token, ''):
                        emoticons_detected = True
                else:
                    if _verify_emoticon(tmp_token, token):
                        new_tokenized_list.append(tmp_token + token)
                        emoticons_detected = True
                    else:
                        new_tokenized_list.append(tmp_token)
                        new_tokenized_list.append(token)
                    #end if
                    break
                #end if
            #end for
        #end if
        i += 1
    #end while
    if not flag:
        return new_tokenized_list
    else:
        return (new_tokenized_list, emoticons_detected)
#end def


def seg_sentences(text):
    # sentences = regex.split(r'[.?!]\s+|\.+\s+', text)
    text = regex.sub(r'\.(?=[^ \W\d])', '. ', text)
    text = regex.sub(r'\!(?=[^ \W\d])', '! ', text)
    text = regex.sub(r'\?(?=[^ \W\d])', '? ', text)

    sentences = sent_tokenize(text)
    new_sentences = list()
    for sentence in sentences:
        tmp_sentence_list = list()
        tokenized = TreebankWordTokenizer().tokenize(sentence)
        tokenized = [SPECIAL_TOKEN[token] if SPECIAL_TOKEN.get(token, '') else token for token in tokenized]

        t, emoticons_detected = _emoticons_detection(tokenized, flag=True)
        if emoticons_detected:
            for i in range(len(t)):
                if (t[i] in EMOTICONS_TOKEN):
                    try:
                        if t[i+1][0].isupper():
                            t[i] = t[i] + '.'
                    except IndexError:
                        pass
                    #end try
                #end if
            #end for
            tmp_sentences = ' '.join(t)
            new_sentences += sent_tokenize(tmp_sentences)
        else:
            new_sentences.append(sentence)

    return new_sentences
#end def

def plot_bar(number_list, title, x_label, y_label, countplot = True):
    sns.set(font_scale = 1.5)

    if countplot:
        fig = sns.countplot(number_list, color = 'c')
    else:
        fig = sns.distplot(number_list, color = 'c', kde = False)

    plt.title(title, loc = 'center', y=1.08, fontsize = 30)
    fig.set_xlabel(x_label)
    fig.set_ylabel(y_label)
    plt.tight_layout()
    saved_path = os.path.join(IMAGES_DIRECTORY, title.lower().replace(' ', '_'))
    fig.get_figure().savefig(saved_path, dpi=200, bbox_inches="tight")
    print('Saved ' + saved_path)
    plt.close()

flatten = lambda l: [item for sublist in l for item in sublist]
