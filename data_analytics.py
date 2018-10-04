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
COUNT = []

def main():
    parser = ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('-i', '--data', type=FileType('r'), nargs='+', metavar='<data>', required=True, help='List of json data files to partition.')
    # parser.add_argument('-i', '--data', type=str)
    A = parser.parse_args()

    # set seed
    np.random.seed(7)

    # load in a pd.df
    df = load_instances(A.data)

    data = []
    import json
    with open('data/CellPhoneReview.json') as f:
        for line in f:
            data.append(json.loads(line))

    # make directory for images
    if not os.path.exists(IMAGES_DIRECTORY):
        os.mkdir(IMAGES_DIRECTORY)

    # ## 3.2.1 get top 10 products
    # print('=' * 50)
    # print('3.2.1 Popular Products and Frequent Reviewers')
    # top_10_products_df = df['asin'].value_counts().head(10).reset_index().rename(columns = {'index': 'productID', 'asin': 'reviewCount'})

    # print('=' * 30)
    # print('Top 10 products:')
    # print('-' * 30)
    # display(top_10_products_df)
    # print('=' * 30)
    # #     productID  reviewCount
    # # 0  B005SUHPO6          836
    # # 1  B0042FV2SI          690
    # # 2  B008OHNZI0          657
    # # 3  B009RXU59C          634
    # # 4  B000S5Q9CA          627
    # # 5  B008DJIIG8          510
    # # 6  B0090YGJ4I          448
    # # 7  B009A5204K          434
    # # 8  B00BT7RAPG          431
    # # 9  B0015RB39O          424


    # ## 3.2.1 get top 10 reviewers
    # top_10_reviewers_df = df['reviewerID'].value_counts().head(10).reset_index().rename(columns = {'index': 'reviewerID', 'reviewerID': 'reviewCount'})

    # print('=' * 30)
    # print('Top 10 reviewers:')
    # print('-' * 30)
    # display(top_10_reviewers_df)
    # print('=' * 30)
    # #        reviewerID  reviewCount
    # # 0  A2NYK9KWFMJV4Y          152
    # # 1  A22CW0ZHY3NJH8          138
    # # 2  A1EVV74UQYVKRY          137
    # # 3  A1ODOGXEYECQQ8          133
    # # 4  A2NOW4U7W3F7RI          132
    # # 5  A36K2N527TXXJN          124
    # # 6  A1UQBFCERIP7VJ          112
    # # 7   A1E1LEVQ9VQNK          109
    # # 8  A18U49406IPPIJ          109
    # # 9   AYB4ELCS5AM8P          107

    # print()

    ## 3.2.2 Sentence segmentation
    print('=' * 50)
    print('3.2.2 Sentence Segmentation')

    print(str(datetime.datetime.now()).split('.')[0] + ': Start processing sentence segmentation')
    df['segmentedSentences'] = df['reviewText'].apply(seg_sentences)
    print(len(COUNT))
    print(Counter(COUNT))
    # df['sentenceCount'] = df['segmentedSentences'].apply(len)
    # print(str(datetime.datetime.now()).split('.')[0] + ': Finish processing sentence segmentation')

    # # plotting for number of sentences
    # plot_bar(df['sentenceCount'], \
    #         title = 'Distribution of Number of Sentences for Each Review', \
    #         x_label = "Sentence Count", y_label = "Review Count", countplot = False)

    # plot_bar(df['sentenceCount'].clip(0, 50), \
    #         title = 'Distribution of Number of Sentences for Each Review (Clipped)', \
    #         x_label = "Sentence Count (Clipped)", y_label = "Review Count", countplot = True)

    # print()

    # ## 3.2.3 Tokenization and Stemming
    # print('=' * 50)
    # print('3.2.3 Tokenization and Stemming')
    # ### No Stemming, with stopwords
    # print('No Stemming, with stopwords:')

    # print(str(datetime.datetime.now()).split('.')[0] + ': Start processing tokenizing')
    # df['tokenizedWord'] = df['segmentedSentences'].apply(lambda sentences: flatten([tokenize(sentence, unique=False, freq=False) for sentence in sentences]))
    # df['wordCount'] = df['tokenizedWord'].apply(len)
    # print(str(datetime.datetime.now()).split('.')[0] + ': Finish processing tokenizing')

    # plot_bar(df['wordCount'], \
    #         title = 'Distribution of Number of Words for Each Review Without Stemming', \
    #         x_label = "Word Count", y_label = "Review Count", countplot = False)
    # plot_bar(df['wordCount'].clip(0, 300), \
    #         title = 'Distribution of Number of Words for Each Review Without Stemming (Clipped)', \
    #         x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    # tokenized_word_list = flatten(df['tokenizedWord'])
    # top_20_words = pd.DataFrame.from_dict(Counter(tokenized_word_list), orient='index').\
    #             reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
    #             sort_values(['Count'], ascending = False).head(20).\
    #             reset_index().drop(columns = ['index'])
    # print('=' * 30)
    # print('Top 20 Words without Stemming')
    # print('-' * 30)
    # display(top_20_words)
    # print('=' * 30)
    # print()

    # ### With Stemming, with stopwords
    # print('With Stemming, with stopwords:')

    # print(str(datetime.datetime.now()).split('.')[0] + ': Start processing tokenizing')
    # stemmer = SnowballStemmer("english")
    # df['stemmedTokenizedWord'] = df['tokenizedWord'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    # df['stemmedWordCount'] = df['stemmedTokenizedWord'].apply(len)
    # print(str(datetime.datetime.now()).split('.')[0] + ': Finish processing tokenizing')

    # plot_bar(df['stemmedWordCount'], \
    #         title = 'Distribution of Number of Words for Each Review With Stemming', \
    #         x_label = "Stemmed Word Count", y_label = "Review Count", countplot = False)
    # plot_bar(df['stemmedWordCount'].clip(0, 300), \
    #         title = 'Distribution of Number of Words for Each Review With Stemming (Clipped)', \
    #         x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    # stemmed_tokenized_word_list = flatten(df['stemmedTokenizedWord'])
    # stemmed_top_20_words = pd.DataFrame.from_dict(Counter(stemmed_tokenized_word_list), orient='index').\
    #             reset_index().rename(columns = {'index': 'Word', 0: 'Count'}).\
    #             sort_values(['Count'], ascending = False).head(20).\
    #             reset_index().drop(columns = ['index'])
    # print('=' * 30)
    # print('Top 20 Words with Stemming')
    # print('-' * 30)
    # display(stemmed_top_20_words)
    # print('=' * 30)
    # print()

    # # ## 3.2.4 POS Tagging
    # print('=' * 50)
    # print('3.2.4 POS Tagging')

    # sentences = pd.Series(flatten(df['segmentedSentences']))
    # print('Total Number of Sentences: ' + str(len(sentences)))

    # random_5_sentences = pd.Series(sentences).sample(5, random_state=5)
    # random_5_df = pd.DataFrame(random_5_sentences, columns = ['sentence']).reset_index().drop(columns = ['index'])
    # random_5_df['tokenizedSentence'] = random_5_df['sentence'].apply(tokenize, unique=False, freq=False, stopwords = False, remove_punc=False, lower=False)
    # random_5_df['posTagged'] = random_5_df['tokenizedSentence'].apply(pos_tag)
    # print('=' * 30)
    # display(random_5_df)
    # print('=' * 30)
#end def


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
    global COUNT 

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
                        COUNT.append(tmp_token)
                        emoticons_detected = True
                else:
                    if _verify_emoticon(tmp_token, token):
                        new_tokenized_list.append(tmp_token + token)
                        COUNT.append(tmp_token + token)
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

if __name__ == '__main__': main()
