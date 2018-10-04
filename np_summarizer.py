from argparse import ArgumentParser, FileType
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import regex as re
import string
import os
import json

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.tokenize.treebank import MacIntyreContractions, TreebankWordTokenizer
from nltk.tokenize.casual import EMOTICON_RE
from nltk import RegexpParser

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
plt.style.use('seaborn-whitegrid')
params = {'figure.figsize': (20,15),
            'savefig.facecolor': 'white'}
plt.rcParams.update(params)

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 10)

DEFAULT_DATA_FILE = "./data/sample_data.json"
IMAGES_DIRECTORY = './images'
DEFAULT_SEED = 42
STOPWORDS = set(stopwords.words('english') + ["'s", "one", "use", "would", "get", "also"]) - {'not', 'no', 'won', 'more', 'above', 'very', 'against', 'again'}

flatten = lambda l: [item for sublist in l for item in sublist]
is_word = lambda token: not(EMOTICON_RE.search(token) or token in string.punctuation or token in STOPWORDS)

class ReviewTokenizer(TreebankWordTokenizer):

    _contractions = MacIntyreContractions()
    CONTRACTIONS = list(map(re.compile, _contractions.CONTRACTIONS2 + _contractions.CONTRACTIONS3))

    PUNCTUATION = [
        (re.compile(r'([,])([^\d])'), r' \1 \2'),
        (re.compile(r'([,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        # Handles the final period
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    def tokenize(self, text):
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



def tokenize(sentence, word_tokenizer = ReviewTokenizer(), stemmer = None, lower = False, remove_punc = False, remove_stopwords = False, remove_emoji = False):

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

    # remove punctuation tokens
    if remove_punc:
        tokens = [token for token in tokens if token not in string.punctuation]

    # stem tokens
    if stemmer:
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def sanitise(text):
    # Append whitespace after punctuations
    for p in '.?!':
        regex = r'\{}(?=[^ \W\d])'.format(p)
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


def extract_NP(posTagged):
    grammar = r"""
        NBAR:
            # Nouns and Adjectives, terminated with Nouns
            {<PRP\$>*<DT.*>*<JJ.*>*<NN.*>+}

        NP:
            {<NBAR><IN><NBAR>}
            {<NBAR>}
        """
    chunker = RegexpParser(grammar)
    ne = []
    chunk = chunker.parse(posTagged)
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.append(' '.join([child[0] for child in tree.leaves()]))
    return ne


data_path = "./data/CellPhoneReview.json"
# data_path = "./data/sample_data.json"
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['overall', 'asin', 'reviewTime', 'reviewerID', 'summary', 'unixReviewTime'])

df['sentences'] = df['reviewText'].apply(segment_sent)
# df['sentenceCount'] = df['sentences'].apply(len)
# df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])


# original_sentences = flatten([[zipped[1]] * zipped[0] for zipped in zip(df['sentenceCount'], df['reviewText'])])
# sentences_df = pd.DataFrame(original_sentences, columns = ['originalSentences']).reset_index().drop(columns = ['index'])
# sentences_df['sentence'] = pd.Series(flatten(df['tokenizedSentences']))
# sentences_df['posTagged'] = sentences_df['sentence'].apply(pos_tag)
# sentences_df['tags'] = sentences_df['posTagged'].apply(lambda posTagged: [tag[1] for tag in posTagged])
# sentences_df['noun_phrases'] = sentences_df['posTagged'].apply(extract_NP)
#
# top_20_noun_phrases = pd.DataFrame.from_dict(Counter(flatten(sentences_df['noun_phrases'])), orient='index').\
#                 reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
#                 sort_values(['Count'], ascending = False).head(20).\
#                 reset_index().drop(columns = ['index'])
# top_20_noun_phrases
# sentences_df.head()
#
# sentences_df.head(10)


def get_emojis(sentences):
    emojis = []
    for sentence in sentences:
        for emoticon in list(EMOTICON_RE.finditer(sentence))[::-1]:
            emojis.append(emoticon.string[(emoticon.span()[0]):])
    return emojis

df['sentenceLen'] = df['sentences'].apply(lambda sentences: [len(sen) for sen in sentences])
df['emojis'] = df['sentences'].apply(get_emojis)
len(flatten(df['emojis']))
set(flatten(df['emojis']))
df[df['sentenceLen'].apply(lambda lens: True if 1 in lens else False)]
sentences_df[sentences_df['sentenceLen'] == 2]
sentences_df[sentences_df['sentenceLen'] == 3]
sentences_df.sort_values(['sentenceLen'])


# set((flatten(sentences_df['tags'])))
# (Adjective | Noun)* (Noun Preposition)? (Adjective | Noun)* Noun

from nltk.tokenize.treebank import TreebankWordTokenizer
pos_tag(TreebankWordTokenizer().tokenize("that's"))


pos_tag(tokenize(segment_sent('that is')[0]))
pos_tag(tokenize(segment_sent("that's")[0]))
pos_tag(tokenize(segment_sent("I'm")[0]))
pos_tag(tokenize(segment_sent("I am")[0]))
pos_tag(tokenize(segment_sent("It is")[0]))
pos_tag(tokenize(segment_sent("It's")[0]))
pos_tag(tokenize(segment_sent("can't")[0]))
posTagged = pos_tag(word_tokenize(sent_tokenize('mr. bean')[0]))


# extract_NN(sent)

# from textblob import TextBlob
# df = df[['reviewText']]
# df['blob'] = df['reviewText'].apply(TextBlob)
# df['sentences'] = df['blob'].apply(lambda blob: blob.sentences)
# df['tokens'] = df['sentences'].apply(lambda sentences: [sentence.tokens for sentence in sentences])
# df['tags'] = df['sentences'].apply(lambda sentences: [sentence.tags for sentence in sentences])
# df['noun_phrases'] = df['blob'].apply(lambda blob: blob.noun_phrases)
# df['sentiment'] = df['sentences'].apply(lambda sentences: [sentence.sentiment.polarity for sentence in sentences])
# df.head()
