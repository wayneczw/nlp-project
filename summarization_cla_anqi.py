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


# get the intersection between two sentences
def getIntersection(s1, s2):
    global title
    s1 = set(s1)
    s2 = set(s2)

    # if the sentences are empty the rank of this will be 0
    if (len(s1) + len(s2)) == 0:
        return 0

    # returning the score of the sentence s1 wrt s2
    return len(s1 & s2)/((len(s1) + len(s2))/2)

def getWMD(s1, s2):
    return word2vec_model.wv.wmdistance(s1, s2)


# create a key for an object from a sentence
def sentenceKey(sentence):
    sentence = re.sub(r'\W+', '', sentence)
    return sentence

def preprocessSentence(sentence):
    stemmer = SnowballStemmer("english")
    return tokenize(sentence, lower = True, stemmer = stemmer, remove_stopwords = True, remove_punc = True)

def rankSentences(sentences, distanceFunction):

    n = len(sentences)

    # Create the sentence adjacency matrix
    values = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            values[i][j] = distanceFunction(sentences[i], sentences[j])

    # create sentence adjacency matrix
    values = np.dot(values, (np.ones((n,n)) - np.eye(n)))
    score = np.sum(values, axis=1)
    sentenceScore = {i:score[i] for i in range(n)}
    print('Finish computing distance.')

    return sentenceScore


# get the best sentence
def getBestSentences(sentences, sentenceScore, limit = 5):

    selected_indixes = sorted(sentenceScore, key=sentenceScore.get, reverse=True)[:limit]
    # selected_indixes = sorted(sentenceScore, key=sentenceScore.get)[:limit]
    return [sentences[i] for i in selected_indixes]


def doSummary(content):
    sentences = segment_sent(content)
    processedSentences = [preprocessSentence(sentence) for sentence in sentences]

    sentenceScore = rankSentences(sentences, getIntersection)
    summary = getBestSentences(sentences, sentenceScore)
    return summary


def summarize_by_cos(content):
    sentences = segment_sent(content)
    processedSentences = [preprocessSentence(sentence) for sentence in sentences]
    processedSentences = [' '.join(sentence) for sentence in processedSentences]

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processedSentences)

    from sklearn.metrics.pairwise import cosine_similarity
    values = cosine_similarity(tfidf_matrix, tfidf_matrix)
    n = values.shape[0]
    values = np.dot(values, (np.ones((n,n)) - np.eye(n)))
    score = np.sum(values, axis=1)
    sentenceScore = {i:score[i] for i in range(n)}

    selected_indixes = sorted(sentenceScore, key=sentenceScore.get, reverse=True)[:5]
    selected_indixes
    return [sentences[i] for i in selected_indixes]



data_path = "./data/CellPhoneReview.json"
# data_path = "./data/sample_data.json"
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['overall', 'reviewTime', 'summary', 'unixReviewTime'])

productId = df['asin'].value_counts().reset_index().iloc[0]['index']
product_df = df.groupby('asin')['reviewText'].sum().reset_index()
product_df = product_df.iloc[:100]
content = product_df[product_df['asin'] == productId].iloc[0]['reviewText']

# summarize_by_cos(content)
# doSummary(content)

product_df['summary'] = product_df['reviewText'].apply(summarize_by_cos)


# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
# from sumy.utils import get_stop_words
# from sumy.nlp.stemmers import Stemmer
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.parsers.plaintext import PlaintextParser
#
# parser = PlaintextParser.from_string(' '.join(segment_sent(content)), Tokenizer('english'))
# stemmer = Stemmer('english')
# summarizer = Summarizer(stemmer)
# summarizer.stop_words = get_stop_words('english')
# for sentence in summarizer(parser.document, 5):
#     print(sentence)


# import gensim
# word2vec_model = gensim.models.Word2Vec(
#     processedSentences,
#     seed=42,
#     workers=10,
#     size=150,
#     min_count=2,
#     window=10)
#
# word2vec_model.train(sentences=processedSentences, total_examples=len(processedSentences), epochs=10)
# word2vec_model.wv.wmdistance(tokenize('nice phone'), tokenize('good phone'))
# word2vec_model.wv.wmdistance(processedSentences[0], processedSentences[1])
#
# word2vec_model.save("word2vec_model1.w2v")
# print("Model saved")
# word2vec_model = gensim.models.Word2Vec.load("word2vec_model1.w2v")
# print("Model loaded")
