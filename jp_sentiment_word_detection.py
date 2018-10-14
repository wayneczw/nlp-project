from argparse import ArgumentParser, FileType
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import regex as re
import string
import os
import json

from nltk import pos_tag,pos_tag_sents
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.tokenize.treebank import MacIntyreContractions, TreebankWordTokenizer
from nltk.tokenize.casual import EMOTICON_RE, _replace_html_entities

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


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
STOPWORDS = set(stopwords.words('english') + ["'ve','s", "one", "use", "would", "get", "also"]) - {'not', 'no', 'won', 'more', 'above', 'very', 'against', 'again'}
PUNCTUATION = string.punctuation + '...'
SPECIAL_TOKEN_DICT = {"n't": 'not'}

flatten = lambda l: [item for sublist in l for item in sublist]
is_word = lambda token: not(EMOTICON_RE.search(token) or token in string.punctuation or token in STOPWORDS)

NEG_SENT_BOUND_RE = re.compile(EMOTICON_RE.pattern +'|'+'|'.join(['\.','\:','\;','\!','\?','\,']))
NEG_WORD_RE = re.compile(r"(?:^(?:never|no|nothing|nowhere|noone|none|not)$)")


def main(data_file, seed):

    # set seed
    np.random.seed(seed)

    # load in a pd.df
    data = [json.loads(line) for file in data_file for line in file]
    df = pd.DataFrame.from_dict(data)

    if not os.path.exists(IMAGES_DIRECTORY):
        os.mkdir(IMAGES_DIRECTORY)

    # 1. Sentence Segmentation
    print(str(datetime.datetime.now()).split('.')[0] + ': Start processing sentence segmentation')
    df['sentences'] = df['reviewText'].apply(segment_sent)

    # 2. Tokenization
    # (1) convert emoticon from regex to text
    # (2) remove stop words
    # (3) convert all the other words in a sentence with negation words to NOT_word

    print(str(datetime.datetime.now()).split('.')[0] + ': Start processing tokenizing')
    df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence, lower = True, remove_punc = True, remove_stopwords = False, convert_neg = True) for sentence in sentences])

    # df['posTagged'] = df['tokenizedSentences'].apply(lambda sentences: pos_tag_sents(sentences))

    df['tokens'] = df['tokenizedSentences'].apply(flatten)
    df['tokens'] = df['tokens'].apply(lambda tokens: list(set(tokens)))
    print(str(datetime.datetime.now()).split('.')[0] + ': Finish processing tokenizing')

    df_positive = df[df['overall'] > 3]
    df_negative = df[df['overall'] < 3]

    words_positive = flatten(df_positive['tokens'])
    words_negative = flatten(df_negative['tokens'])

    df_top1000_words_positive = pd.DataFrame.from_dict(Counter(words_positive), orient='index').\
                reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
                sort_values(['Count'], ascending = False).head(1000).\
                reset_index().drop(columns = ['index'])
    df_top1000_words_negative = pd.DataFrame.from_dict(Counter(words_negative), orient='index').\
                reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
                sort_values(['Count'], ascending = False).head(1000).\
                reset_index().drop(columns = ['index'])

    top1000_words_positive_set = set(df_top1000_words_positive['Word'])
    top1000_words_negative_set = set(df_top1000_words_negative['Word'])
    common_words = set.intersection(top1000_words_positive_set,top1000_words_negative_set)

    top20_words_positive = df_top1000_words_positive[~df_top1000_words_positive['Word'].isin(common_words)].nlargest(20,'Count')
    top20_words_negative = df_top1000_words_negative[~df_top1000_words_negative['Word'].isin(common_words)].nlargest(20,'Count')

    print(str(datetime.datetime.now()).split('.')[0] + ': top 20 positive words')
    print(top20_words_positive)

    #     2018-10-05 14:59:56: top 20 positive words
    #              Word  Count
    # 222  highly        6946
    # 311  protects      4823
    # 366  sturdy        4003
    # 373  durable       3910
    # 404  loves         3545
    # 405  tablet        3544
    # 446  amazing       3254
    # 452  recommended   3208
    # 459  protected     3143
    # 460  pleased       3143
    # 462  provides      3141
    # 464  handy         3121
    # 472  allows        3077
    # 475  neg_problems  3068
    # 478  bulk          3037
    # 481  provided      3001
    # 492  gives         2921
    # 499  nicely        2879
    # 502  led           2855
    # 525  travel        2664

    print(str(datetime.datetime.now()).split('.')[0] + ': top 20 negative words')
    print(top20_words_negative)

    # 2018-10-05 14:59:57: top 20 negative words
    #               Word  Count
    # 200  disappointed   1192
    # 211  waste          1115
    # 228  return         1061
    # 231  neg_buy        1049
    # 238  neg_recommend  1028
    # 272  neg_worth      914
    # 323  poor           771
    # 327  returned       765
    # 334  stopped        738
    # 370  apart          657
    # 374  neg_again      652
    # 386  unfortunately  631
    # 395  useless        603
    # 426  send           562
    # 433  refund         545
    # 437  fell           543
    # 458  broken         514
    # 476  flimsy         489
    # 478  horrible       489
    # 497  neg_money      467
#end def

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
    '''
        During tokenization:
        if the sentence contains negation words, convert all the other words into ("NOT_"+word) until NEG_SENTENCE_BOUNDARY_LIST
    '''
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
#end def

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

def print_header(text, width = 30, char = '='):
    print('\n' + '=' * width)
    print(text)
    print('=' * width)

if __name__ == '__main__':
    parser = ArgumentParser(description = """
        CZ4045 NLP Project: Product Review Data Analysis and Processing. Use '--help' to list available arguments.
    """)
    parser.add_argument('-i', '--data', type = FileType('r'), nargs = '+', metavar = '<data>',
        required = False, help = 'Product review data in JSON format', default = DEFAULT_DATA_FILE)
    parser.add_argument('-s', '--seed', type = int, nargs = '+', metavar = '<seed>',
        required = False, help = 'Seed to be used for pseudo randomisations', default = DEFAULT_SEED)
    args = parser.parse_args()

    main(args.data, args.seed)
