from argparse import ArgumentParser, FileType
from collections import Counter, OrderedDict
import argparse
import math
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

def main(data_file, seed):

    # set seed
    np.random.seed(seed)

    # load in a pd.df
    data = [json.loads(line) for line in data_file]
    df = pd.DataFrame.from_dict(data)

    # make directory for images
    if not os.path.exists(IMAGES_DIRECTORY):
        os.mkdir(IMAGES_DIRECTORY)
    
    stemmer = SnowballStemmer("english")
    # df = df.sample(100)
    df['sentences'] = df['reviewText'].apply(segment_sent)
    df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence, stemmer = stemmer, lower = True, remove_punc = True, remove_stopwords = True, remove_emoji = False, convert_neg=True) for sentence in sentences])
    df['tokens'] = df['tokenizedSentences'].apply(flatten)
    df['words'] = df['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
    df['words'] = df['words'].apply(lambda tokens: [token for token in tokens if is_word(token)])
 
    it_score_dict = it_score(df)  
    probabilistic_score_dict = probabilistic_score(df)
    word_score_dict = dict()
    for word in probabilistic_score_dict.keys():
        word_score_dict[word] = (probabilistic_score_dict[word] + it_score_dict[word]) / 2

    orderd_word_score_dict = OrderedDict(sorted(word_score_dict.items(), key=lambda t: t[1], reverse=True))
    pd.DataFrame.from_dict(orderd_word_score_dict, orient="index").to_csv("./rep_words/rep_words.csv")
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
    # Append whitespace after punctuations, except .com
    for p in '.?!':
        regex = r'(?<=[^\d]+)\{}(?=[^ \W])(?!com)'.format(p)
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
            {<PREFIXEDNOUN> <PP>*}
            {<PRP>}

        """
    chunker = RegexpParser(grammar)
    ne = []
    chunk = chunker.parse(posTagged)
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.append(' '.join([child[0] for child in tree.leaves()]))
    return ne


def probabilistic_score(df):
    def _flatten_tokens(token_series):
        return [word for token_list in token_series for word in token_list]

    def _nwr(nwh, nwl, gamma=4):
        return gamma * nwh + nwl

    def _word_give_tag(nwr, sum_nwr, k):
        return ((nwr  / sum_nwr) + 1) / (k + 1)

    pos_rev5 = _flatten_tokens(df.loc[df['overall'] == 5]['words'])
    pos_rev4 = _flatten_tokens(df.loc[df['overall'] == 4]['words'])
    neg_rev2 = _flatten_tokens(df.loc[df['overall'] == 2]['words'])
    neg_rev1 = _flatten_tokens(df.loc[df['overall'] == 1]['words'])

    words = set()
    pos_words = set()
    neg_words = set()

    pos_count5 = Counter()
    for word in pos_rev5:
        pos_count5[word] += 1
        words.add(word)
        pos_words.add(word)

    pos_count4 = Counter()
    for word in pos_rev4:
        pos_count4[word] += 1
        words.add(word)
        pos_words.add(word)

    neg_count2 = Counter()
    for word in neg_rev2:
        neg_count2[word] += 1
        words.add(word)
        neg_words.add(word)

    neg_count1 = Counter()
    for word in neg_rev1:
        neg_count1[word] += 1
        words.add(word)
        neg_words.add(word)

    kpos = len(pos_words)  # kpos
    kneg = len(neg_words)  # kneg 
    kdic = len(words)  # kdic

    # compute nwr dict
    pos_nwr_dict = dict()
    sum_pos_nwr = 0
    for word in pos_words:
        nwh = pos_count5.get(word, 0)
        nwl = pos_count4.get(word, 0)
        nwr = _nwr(nwh=nwh, nwl=nwl)
        sum_pos_nwr += nwr
        pos_nwr_dict[word] = nwr

    neg_nwr_dict = dict()
    sum_neg_nwr = 0
    for word in neg_words:
        nwh = neg_count1.get(word, 0)
        nwl = neg_count2.get(word, 0)
        nwr = _nwr(nwh=nwh, nwl=nwl)
        sum_neg_nwr += nwr
        neg_nwr_dict[word] = nwr


    p_pos = sum_pos_nwr # P(pos)
    p_neg = sum_neg_nwr  # P(neg)

    p_word_pos_dict = dict()
    p_word_neg_dict = dict()
    for word in words:
        p_word_pos_dict[word] = _word_give_tag(pos_nwr_dict.get(word, 0), sum_pos_nwr, kpos)
        p_word_neg_dict[word] = _word_give_tag(neg_nwr_dict.get(word, 0), sum_neg_nwr, kneg)

    p_pos_word_dict = dict()
    p_neg_word_dict = dict()
    for word in words:
        p_pos_word_dict[word] = (p_pos * p_word_pos_dict[word]) / (pos_nwr_dict.get(word, 0) + neg_nwr_dict.get(word, 0))
        p_neg_word_dict[word] = (p_neg * p_word_neg_dict[word]) / (pos_nwr_dict.get(word, 0) + neg_nwr_dict.get(word, 0))

    score_dict = dict()
    for word in words:
        score_dict[word] = round(p_pos_word_dict[word] - p_neg_word_dict[word], 3)

    return OrderedDict(sorted(score_dict.items(), key=lambda t: t[1], reverse=True))
#end def


def it_score(df):
    nNeg = len(df[df['overall'].isin([1,2])])
    nPos = len(df[df['overall'].isin([4,5])])
    nN = nNeg + nPos
    df = df.loc[df['overall'] != 3]

    lexicons = set()
    rtf_dict1 = Counter()
    rtf_dict2 = Counter()
    rtf_dict4 = Counter()
    rtf_dict5 = Counter()
    df_count = Counter()

    for index, row in df.iterrows():
        counter = Counter(row['words'])
        reviewLength = sum(counter.values())
        for key in counter.keys():
            lexicons.add(key)
            df_count[key] += 1
            if row['overall'] == 1:
                rtf_dict1[key] += (counter[key]/reviewLength)
            elif row['overall'] == 2:
                rtf_dict2[key] += (counter[key]/reviewLength)
            elif row['overall'] == 4:
                rtf_dict4[key] += (counter[key]/reviewLength)
            elif row['overall'] == 5:
                rtf_dict5[key] += (counter[key]/reviewLength)

    score_it_dict = dict()
    for word in lexicons:
        pos_word = 4 * (rtf_dict5.get(word, 0)/nPos) * nN + rtf_dict4.get(word, 0)/nPos * nN
        neg_word = 4 * (rtf_dict1.get(word, 0)/nNeg) * nN + rtf_dict2.get(word, 0)/nNeg * nN
        idf_word = math.log(nN/df_count.get(word))
        score_it_dict[word] = round((pos_word - neg_word) * idf_word, 3)

    return OrderedDict(sorted(score_it_dict.items(), key=lambda t: t[1], reverse=True))
#end def


def print_header(text, width = 30, char = '='):
    print('\n' + char * width)
    print(text)
    print(char * width)

if __name__ == '__main__':
    parser = ArgumentParser(description = """
        CZ4045 NLP Project: Product Review Data Analysis and Processing. Use '--help' to list available arguments.
    """)
    parser.add_argument('-i', '--data', type = FileType('r'), metavar = '<data>',
        required = False, help = 'Product review data in JSON format', default = DEFAULT_DATA_FILE)
    parser.add_argument('-s', '--seed', type = int, metavar = '<seed>',
        required = False, help = 'Seed to be used for pseudo randomisations', default = DEFAULT_SEED)
    args = parser.parse_args()

    main(args.data, args.seed)
