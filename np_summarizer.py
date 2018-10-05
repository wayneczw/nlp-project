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
from nltk.tokenize.casual import EMOTICON_RE, _replace_html_entities
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
# df = df.sample(10000)

df['sentences'] = df['reviewText'].apply(segment_sent)
# df['sentenceCount'] = df['sentences'].apply(len)
df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])


# original_sentences = flatten([[zipped[1]] * zipped[0] for zipped in zip(df['sentenceCount'], df['reviewText'])])
# sentences_df = pd.DataFrame(original_sentences, columns = ['originalSentences']).reset_index().drop(columns = ['index'])
# sentences_df['sentence'] = pd.Series(flatten(df['tokenizedSentences']))
sentences_df = pd.DataFrame(pd.Series(flatten(df['tokenizedSentences'])), columns = ['sentence']).reset_index().drop(columns = ['index'])

sentences_df[sentences_df['sentence'].apply(lambda sentence: True if '&' in sentence else False)]
sentences_df['posTagged'] = sentences_df['sentence'].apply(pos_tag)
sentences_df['tags'] = sentences_df['posTagged'].apply(lambda posTagged: [tag[1] for tag in posTagged])
sentences_df['noun_phrases'] = sentences_df['posTagged'].apply(extract_NP)

top_20_noun_phrases = pd.DataFrame.from_dict(Counter(flatten(sentences_df['noun_phrases'])), orient='index').\
                reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
                sort_values(['Count'], ascending = False).head(20).\
                reset_index().drop(columns = ['index'])
top_20_noun_phrases
sentences_df.head()

sentences_df.head(10)

class EmojiTokenizer(TreebankWordTokenizer):

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

    emojis = []

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
            self.emojis.append(emoticon.group())
            pos = emoticon.span()[0]
            if text[pos - 1] != ' ':
                text = text[:pos] + ' ' + text[pos:]

        return text.split()

_replace_html_entities('&#34;')
tokenize(segment_sent('&#34;100')[0])

tokenizer = EmojiTokenizer()
for sentence in flatten(df['sentences']):
    tokenize(sentence, word_tokenizer = tokenizer)
emojis = tokenizer.emojis

clarence = set(tokenizer.emojis)

zhiwei = {':)': 2520, ':(': 808, ':*': 664, ';)': 427, ':&': 357, ':D': 163, ':[': 152, '=)': 149, ':/': 100, ':>': 57, ':3': 46, ':$': 46, ':P': 36, ':o': 24, '8)': 18, ':-*': 18, ':#': 15, ':o)': 15, ':p': 15, '=]': 14, ':]': 10, ':-))': 9, '=3': 8, ':O': 7, ';D': 7, ':-3': 6, 'D:': 5, ":'(": 5, ':S': 5, '*)': 5, ':-}': 4, ':\\': 4, ':{': 3, ';]': 3, ':c': 3, ':b': 3, ':|': 2, ':^)': 2, '>:(': 2, '3:)': 2, '>:[': 1, ':X': 1, ':L': 1, ':@': 1, ':-0': 1, ':}': 1, '>:)': 1, ":')": 1, '>_>': 1, 'DX': 1}
zhiwei_set = set(zhiwei.keys())
clarence = [(':)', 2561), (':(', 667), (':P', 521), ('/8', 512), (':-)', 493), ('):', 482), ('p:', 384), (':D', 345), ('(8', 204), ('8/', 190), (':/', 162), ('=)', 152), ('(:', 143), ('P8', 134), (':-(', 107), ('8)', 79), (':=', 77), ('8:', 77), (':[', 71), (':-D', 57), ('do:', 50), (':-P', 49), ('::', 47), (':p', 42), ('8-p', 31), (']:', 29), ('|8', 27), ('=(', 24), (':-/', 24), ('8p', 24), ('8-P', 19), (':o)', 15), (')8', 14), ('=]', 14), ('D8', 13), (':-p', 13), ('=p', 12), (')=', 11),      ('=P', 11), ('=D', 10), (':]', 9), ('/:', 8), (':d', 7), ('8P', 7), ('=/', 7), ('=-)', 7), ('8D', 7), (':8', 7), ('(=', 6), (':*P', 6), ('=[', 6), (':-d', 6), ('<3', 6), ('D:', 6), (':\\', 5), ('=d', 5), ('8d', 5), ('(-:', 5), (":'(", 5), (':O)', 4), ('>:(', 4), (':-}', 4), (':*)', 3), (':-{', 3), (':*D', 3), ('=o)', 3), (':}', 3), (':-\\', 3), ('=|', 2), (':Op', 2), (':|', 2), ('8-)', 2), ('=}', 2), ('=\\', 2), (':*(', 2), ('8-d', 2), (':-8', 2), ('>:[', 1), (':OP', 1), ('>:)', 1), ('(-8', 1), ('=-[', 1), ('>:|', 1), ('[:', 1), (':*[', 1), (']=', 1), (':{', 1), (':O/', 1), ('[=', 1), (':o/', 1), ('=-D', 1), (":')", 1), (')=>', 1), (':-|', 1), ('/=', 1), ('\\8', 1), (':*d', 1), ('8oP', 1)]
clarence_set = [pair[0] for pair in clarence]

{emoji: zhiwei[emoji] for emoji in zhiwei_set.difference(clarence_set)}
len(zhiwei.difference(clarence))
len(clarence.difference(zhiwei))
len(clarence.intersection(zhiwei))
clarence
zhiwei
print(Counter(emojis).most_common())

df['sentenceLen'] = df['sentences'].apply(lambda sentences: [len(sen) for sen in sentences])

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
