from argparse import ArgumentParser, FileType
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import regex as re
import string
import os
import json
import datetime

from nltk import pos_tag, RegexpParser
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.tokenize.treebank import MacIntyreContractions, TreebankWordTokenizer
from nltk.tokenize.casual import EMOTICON_RE, _replace_html_entities

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
plt.style.use('seaborn-whitegrid')
params = {'figure.figsize': (15,12),
            'savefig.facecolor': 'white'}
plt.rcParams.update(params)

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 10)

DEFAULT_DATA_FILE = "./data/sample_data.json"
IMAGES_DIRECTORY = './images'
DEFAULT_SEED = 42
STOPWORDS = set(stopwords.words('english') + ["'ve", "'d", "'s", "one", "use", "would", "get", "also"]) - {'not', 'no', 'won', 'more', 'above', 'very', 'against', 'again'}
PUNCTUATION = string.punctuation + '...'
SPECIAL_TOKEN_DICT = {"n't": 'not'}
boundary_punc = '.:;!?,'
NEG_SENT_BOUND_RE = re.compile(EMOTICON_RE.pattern + '|' + '|'.join([re.escape(punc) for punc in boundary_punc]))
NEG_WORD_RE = re.compile(r"(?:^(?:never|no|nothing|nowhere|noone|none|not)$)")

flatten = lambda l: [item for sublist in l for item in sublist]
is_word = lambda token: not(EMOTICON_RE.search(token) or token in string.punctuation or token in STOPWORDS)

class ReviewTokenizer(TreebankWordTokenizer):

    _contractions = MacIntyreContractions()
    CONTRACTIONS = list(map(re.compile, _contractions.CONTRACTIONS2 + _contractions.CONTRACTIONS3))

    PUNCTUATION = [
        (re.compile(r'([,])([^\d])'), r' \1 \2'),
        (re.compile(r'([,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#&]'), r' \g<0> '),
        # Handles the final period
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    def tokenize(self, text):

        text = _replace_html_entities(text)

        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        text = " " + text + " "

        # split contractions
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)
        for regexp in self.CONTRACTIONS:
            text = regexp.sub(r' \1 \2 ', text)

        # handle emojis
        for emoticon in list(EMOTICON_RE.finditer(text))[::-1]:
            pos = emoticon.span()[0]
            if text[pos - 1] != ' ':
                text = text[:pos] + ' ' + text[pos:]

        return text.split()

def tokenize(sentence, word_tokenizer = ReviewTokenizer(), stemmer = None, lower = False, remove_punc = False, remove_stopwords = False, remove_emoji = False, convert_neg = False):

    tokens = word_tokenizer.tokenize(sentence)

    # convert tokens to lowercase
    if lower:
        tokens = [token.lower() for token in tokens]

    # remove stopword tokens
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    # remove emoji tokens
    if remove_emoji:
        tokens = [token for token in tokens if not EMOTICON_RE.search(token)]

    # NEG_SENTENCE_BOUNDARY_LIST
    if convert_neg == True:
        tokens = _convert_neg(tokens)

    # remove punctuation tokens
    if remove_punc:
        tokens = [token for token in tokens if token not in PUNCTUATION]

    # stem tokens
    if stemmer:
        tokens = [stemmer.stem(token) for token in tokens]

    tokens = [SPECIAL_TOKEN_DICT[token] if SPECIAL_TOKEN_DICT.get(token, '') else token for token in tokens]

    return tokens

def _convert_neg(tokens):
    '''
        convert word to NEGATIVEword if there are negation words
        NEG_WORD_LIST,NEG_SENTENCE_BOUNDARY_LIST
    '''
    new_tokenized_list = list()
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        new_tokenized_list.append(token)
        # e.g. if token is negative word, the following words should be negated
        if NEG_WORD_RE.match(token):
            i += 1
            while (i < n) :
                token = tokens[i]
                if not NEG_SENT_BOUND_RE.match(token):
                    new_tokenized_list.append('neg_'+token)
                else:
                    # end of negation
                    new_tokenized_list.append(token)
                    break
                i += 1
            #end while
        i += 1
    #end while
    return new_tokenized_list

def sanitise(text):

    regex_list = [
        # Append whitespace after punctuations, except .com
        r'(?<=[^\d]+)\{}(?=[^ \W])(?!com)',
        # Reduce repeated punctuations
        r'(\s*\{}\s*)+'
    ]

    for regex_format in regex_list:
        for p in '.?!':
            regex = regex_format.format(p)
            text = re.sub(regex, p + ' ', text)

    return text

def segment_sent(text, emoji_tokenizer = TweetTokenizer()):
    text = sanitise(text)
    sentences = []
    for sentence in sent_tokenize(text):
        if EMOTICON_RE.search(sentence):
            new_sentences = []
            tokens = emoji_tokenizer.tokenize(sentence)
            new_sentence = []
            for token in tokens:
                new_sentence.append(token)
                if EMOTICON_RE.search(token) or token in '.?!':
                    new_sentences.append(' '.join(new_sentence))
                    new_sentence = []
            if new_sentence:
                new_sentences.append(' '.join(new_sentence))
            sentences += new_sentences
        else:
            sentences.append(sentence)

    if len(sentences) != 0:
        if sentences[-1] in ['.', '!', '?']:
            sentences[-2] = sentences[-2] + sentences[-1]
            sentences = sentences[:-1]
    return sentences

def plot_aspect_scores(aspect_score_df, x, y, title, x_label, y_label, showfliers=True):
    sns.set(font_scale = 1.5)
    fig = sns.boxplot(x=x, y=y, data=aspect_score_df, showfliers=showfliers)

    plt.title(title, loc = 'center', y=1.08, fontsize = 30)
    fig.set(ylim=(0, 5))
    fig.set_xlabel(x_label)
    fig.set_ylabel(y_label)
    plt.tight_layout()
    saved_path = os.path.join(IMAGES_DIRECTORY, title.lower().replace(' ', '_'))
    if not os.path.exists(IMAGES_DIRECTORY):
        os.makedirs(IMAGES_DIRECTORY)
    fig.get_figure().savefig(saved_path, dpi=200, bbox_inches="tight")
    print('{} has been saved to {}'.format(title, saved_path))
    plt.close()

print('Thanks for using this aspects extraction application.')
productID = input("What is the ID of the product that you want to extract aspects from?\n")


data_path = "./data/CellPhoneReview.json"
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['overall', 'reviewTime', 'summary', 'unixReviewTime'])

while productID not in list(df['asin']):
    print('Product {} cannot be found in our database. Please input a valid product ID, such as "B005SUHPO6", "B0042FV2SI", "B008OHNZI0" or "B00AYNRLFA"'.format(productID))
    productID = input("Please enter a valid product ID.\n")


print('Start processing Aspect Extraction for product {}.'.format(productID))
print('Please wait for a while as this might take 2-3 min.')

implicit_aspect_polarity_df = pd.read_csv('implicit_aspect_polarity.csv')
aspects = list(implicit_aspect_polarity_df['aspect'].unique())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
implicit_aspect_polarity_df['polarity_intense_scaled'] = scaler.fit_transform(implicit_aspect_polarity_df[['polarity_intense']]) * 5

df = df[df['asin'] == productID]
df['sentences'] = df['reviewText'].apply(segment_sent)
df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])
df['tokens'] = df['tokenizedSentences'].apply(flatten)
df['words'] = df['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
df['words'] = df['words'].apply(lambda tokens: [token for token in tokens if is_word(token)])

for aspect in aspects:
    df[aspect] = df['words'].apply(lambda words: [word for word in words if word in list(implicit_aspect_polarity_df[implicit_aspect_polarity_df['aspect'] == aspect]['implicit'])])

product_df = df.groupby('asin')[aspects].sum().reset_index()

for aspect in aspects:
    aspect_df = implicit_aspect_polarity_df[implicit_aspect_polarity_df['aspect'] == aspect]
    aspect_dict = dict(zip(aspect_df['implicit'], aspect_df['polarity_intense_scaled']))
    product_df[aspect + '_scores'] = product_df[aspect].apply(lambda aspect: [aspect_dict[word] for word in aspect])

aspect_list = []
aspect_score_list = []
aspect_word_list = []
for aspect in aspects:
    aspects_words = product_df.iloc[0].to_dict()[aspect]
    aspects_scores = product_df.iloc[0].to_dict()[aspect + '_scores']
    for i in range(len(aspects_words)):
        aspect_list.append(aspect)
        aspect_score_list.append(aspects_scores[i])
        aspect_word_list.append(aspects_words[i])
aspect_score_df = pd.DataFrame(data = {'aspect': aspect_list, 'word': aspect_word_list, 'score': aspect_score_list})

plot_aspect_scores(aspect_score_df, 'aspect', 'score', title = 'Aspect Scores for ' + productID, x_label = 'aspect', y_label = 'score', showfliers=False)
