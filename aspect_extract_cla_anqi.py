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
params = {'figure.figsize': (20,15),
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

def plot_bar_overlap(df, columns, title, x_label, y_label, countplot = True):
    sns.set(font_scale = 1.5)

    fig, ax = plt.subplots()
    for col in columns:
        if countplot:
            sns.countplot(df[col], ax=ax, label = col)
        else:
            sns.distplot(df[col], ax=ax, kde=False, label = col)

    plt.legend(loc='upper right')
    plt.title(title, loc = 'center', y=1.08, fontsize = 30)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    saved_path = os.path.join(IMAGES_DIRECTORY, title.lower().replace(' ', '_'))
    fig.savefig(saved_path, dpi=200, bbox_inches="tight")
    print('Saved ' + saved_path)
    plt.close()


def extract_NP(posTagged):
    grammar = r"""

        ADJ:
            {<RB.*>? <JJ.* | VBG>}

        ADJLIST:
            {<ADJ> (<CC>? <,>? <ADJ>)*}

        ADJNOUN:
            {<ADJLIST>? <NN.*>+}

        PREFIXEDNOUN:
            {<DT|PRP\$>? (<ADJNOUN> <POS>)* <ADJNOUN>}

        PP:
            {<IN><PREFIXEDNOUN>}

        NP:
            {<PREFIXEDNOUN> (<PP>)*}
            {<PRP>}

        """
    chunker = RegexpParser(grammar)
    ne = []
    chunk = chunker.parse(posTagged)
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.append(' '.join([child[0] for child in tree.leaves()]))
    return ne

def preprocessSentence(sentence):
    return tokenize(sentence, lower = True, remove_stopwords = True, remove_punc = True)



data_path = "./data/CellPhoneReview.json"
data_path = "./data/sample_data.json"
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['overall', 'reviewTime', 'summary', 'unixReviewTime'])

df['sentences'] = df['reviewText'].apply(segment_sent)
df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])
df['cleanedTokenizedSentences'] = df['sentences'].apply(lambda sentences: [preprocessSentence(sentence) for sentence in sentences])
cleanedTokenizedSentences = flatten(df['cleanedTokenizedSentences'])

df['posTagged'] = df['tokenizedSentences'].apply(lambda tokenizedSentences: [pos_tag(sentence) for sentence in tokenizedSentences])
df['nounPhrases'] = df['posTagged'].apply(lambda posTagged: [np.lower() for np in flatten([extract_NP(tag) for tag in posTagged])])
df['uniqueNounPhrases'] = df['nounPhrases'].apply(set).apply(list)
