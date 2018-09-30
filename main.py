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

def main(data_file, seed):

    # set seed
    np.random.seed(seed)

    # load in a pd.df
    data = [json.loads(line) for line in data_file]
    df = pd.DataFrame.from_dict(data)

    # make directory for images
    if not os.path.exists(IMAGES_DIRECTORY):
        os.mkdir(IMAGES_DIRECTORY)

    print_header('3.2.1 Popular Products and Frequent Reviewers', 50)

    ## 3.2.1 get top 10 products
    top_10_products = df['asin'].value_counts().head(10).reset_index().rename(columns = {'index': 'productID', 'asin': 'reviewCount'})
    print_header('Top 10 products', char = '-')
    print(top_10_products)

    #     productID  reviewCount
    # 0  B005SUHPO6          836
    # 1  B0042FV2SI          690
    # 2  B008OHNZI0          657
    # 3  B009RXU59C          634
    # 4  B000S5Q9CA          627
    # 5  B008DJIIG8          510
    # 6  B0090YGJ4I          448
    # 7  B009A5204K          434
    # 8  B00BT7RAPG          431
    # 9  B0015RB39O          424


    ## 3.2.1 get top 10 reviewers
    top_10_reviewers = df['reviewerID'].value_counts().head(10).reset_index().rename(columns = {'index': 'reviewerID', 'reviewerID': 'reviewCount'})
    print_header('Top 10 reviewers', char = '-')
    print(top_10_reviewers)

    #        reviewerID  reviewCount
    # 0  A2NYK9KWFMJV4Y          152
    # 1  A22CW0ZHY3NJH8          138
    # 2  A1EVV74UQYVKRY          137
    # 3  A1ODOGXEYECQQ8          133
    # 4  A2NOW4U7W3F7RI          132
    # 5  A36K2N527TXXJN          124
    # 6  A1UQBFCERIP7VJ          112
    # 7   A1E1LEVQ9VQNK          109
    # 8  A18U49406IPPIJ          109
    # 9   AYB4ELCS5AM8P          107

    ## 3.2.2 Sentence segmentation
    print_header('3.2.2 Sentence Segmentation', 50)

    df['sentences'] = df['reviewText'].apply(segment_sent)
    df['sentenceCount'] = df['sentences'].apply(len)

    # plotting for number of sentences
    plot_bar(df['sentenceCount'], \
            title = 'Distribution of Number of Sentences for Each Review', \
            x_label = "Sentence Count", y_label = "Review Count", countplot = False)

    plot_bar(df['sentenceCount'].clip(0, 50), \
            title = 'Distribution of Number of Sentences for Each Review (Clipped)', \
            x_label = "Sentence Count (Clipped)", y_label = "Review Count", countplot = True)

    ## 3.2.3 Tokenization and Stemming
    print_header('3.2.3 Tokenization and Stemming', 50)

    df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])
    df['tokens'] = df['tokenizedSentences'].apply(flatten)

    ### No Stemming
    print_header('No Stemming', char = '-')
    df['words'] = df['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
    df['words'] = df['words'].apply(lambda tokens: [token for token in tokens if is_word(token)])
    df['wordCount'] = df['words'].apply(len)

    # token = {normal_word, emoji, stopword, punctuation}
    # word = {normal_word, emoji}

    plot_bar(df['wordCount'], title = 'Distribution of Number of Words for Each Review Without Stemming',
            x_label = "Word Count", y_label = "Review Count", countplot = False)
    plot_bar(df['wordCount'].clip(0, 300), title = 'Distribution of Number of Words for Each Review Without Stemming (Clipped)',
            x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    words = flatten(df['words'])
    top_20_words = pd.DataFrame.from_dict(Counter(words), orient='index').\
                reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
                sort_values(['Count'], ascending = False).head(20).\
                reset_index().drop(columns = ['index'])

    print_header('Top 20 Words Without Stemming', char = '-')
    print(top_20_words)

    ### With Stemming
    print_header('With Stemming', char = '-')
    stemmer = SnowballStemmer("english")
    df['stemmedWords'] = df['words'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    df['stemmedWordCount'] = df['stemmedWords'].apply(len)

    plot_bar(df['stemmedWordCount'], \
            title = 'Distribution of Number of Words for Each Review With Stemming', \
            x_label = "Stemmed Word Count", y_label = "Review Count", countplot = False)
    plot_bar(df['stemmedWordCount'].clip(0, 300), \
            title = 'Distribution of Number of Words for Each Review With Stemming (Clipped)', \
            x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    stemmed_words = flatten(df['stemmedWords'])
    top_20_stemmed_words = pd.DataFrame.from_dict(Counter(stemmed_words), orient='index').\
                reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
                sort_values(['Count'], ascending = False).head(20).\
                reset_index().drop(columns = ['index'])

    print_header('Top 20 Words with Stemming', char = '-')
    print(top_20_stemmed_words)

    print_header('3.2.4 POS Tagging', 50)

    tokenized_sentences = pd.Series(flatten(df['tokenizedSentences']))
    print('Total Number of Sentences: ' + str(len(tokenized_sentences)))

    random_5_sentences = tokenized_sentences.sample(5, random_state = seed)
    random_5_df = pd.DataFrame(random_5_sentences, columns = ['sentence']).reset_index().drop(columns = ['index'])
    random_5_df['posTagged'] = random_5_df['sentence'].apply(pos_tag)
    print('=' * 30)
    print(random_5_df)
    print('=' * 30)

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
