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
import random
import regex
import string
from utils import load_instances, load_dictionary_from_file, _process_regex_dict

logger = logging.getLogger(__name__)
EMOTICONS_REGEX = _process_regex_dict(load_dictionary_from_file('./separators.yaml'), regex_escape=True)
EMOTICONS_TOKEN = _process_regex_dict(load_dictionary_from_file('./separators.yaml'))
STOPWORDS = set(stopwords.words('english') + ["'s", "one", "use", "would", "get", "also"])
SPECIAL_TOKEN = {"n't": 'not'}

def main():
    parser = ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('-i', '--data', type=FileType('r'), nargs='+', metavar='<data>', required=True, help='List of json data files to partition.')
    # parser.add_argument('-i', '--data', type=str)
    A = parser.parse_args()

    # set seed
    np.random.seed(7)

    # pandas settings
    pd.set_option('display.max_columns', 10)

    # load in a pd.df
    df = load_instances(A.data)

    ## 3.2.1 get top 10 products
    n = 10
    top_10_products_df = df.groupby(['asin'])['asin'].agg(
        {"count": len}).sort_values(
        "count", ascending=False).head(n).reset_index()

    display(top_10_products_df)
    print('==========================')
    #          asin  count
    # 0  B005SUHPO6    836
    # 1  B0042FV2SI    690
    # 2  B008OHNZI0    657
    # 3  B009RXU59C    634
    # 4  B000S5Q9CA    627
    # 5  B008DJIIG8    510
    # 6  B0090YGJ4I    448
    # 7  B009A5204K    434
    # 8  B00BT7RAPG    431
    # 9  B0015RB39O    424

    ## 3.2.1 get top 10 reviewers
    n = 10
    top_10_reviewers_df = df.groupby(['reviewerID'])['reviewerID'].agg(
        {"count": len}).sort_values(
        "count", ascending=False).head(n).reset_index()

    display(top_10_reviewers_df)
    print('==========================')
    #        reviewerID  count
    # 0  A2NYK9KWFMJV4Y    152
    # 1  A22CW0ZHY3NJH8    138
    # 2  A1EVV74UQYVKRY    137
    # 3  A1ODOGXEYECQQ8    133
    # 4  A2NOW4U7W3F7RI    132
    # 5  A36K2N527TXXJN    124
    # 6  A1UQBFCERIP7VJ    112
    # 7   A1E1LEVQ9VQNK    109
    # 8  A18U49406IPPIJ    109
    # 9   AYB4ELCS5AM8P    107

    ## 3.2.2 Sentence segmentation
    df['numSentences'] = df['reviewText'].apply(seg_sentences)
    # display(df['numSentences'])
    # print('==========================')

    ## 3.2.3 Tokenization and Stemming
    ### No Stemming, with stopwords
    df['numTokenized'] = df['reviewText'].apply(tokenize)
    top_20_words_dict = top_n_words(df['reviewText'], n=20)    
    print('Top 20 Words before Stemming')
    print('----------------------------')
    for word, count in top_20_words_dict.items():
        print('|{:20s}| {:6d}|'.format(word, count)) 
    print('----------------------------')
    print()

    ### With Stemming, with stopwords
    df['numTokenizedAndStemmed'] = df['reviewText'].apply(tokenize_and_stem)
    top_20_words_stemmed_dict = top_n_words(df['reviewText'], stem=True, n=20)    
    print('Top 20 Words after Stemming')
    print('----------------------------')    
    for word, count in top_20_words_stemmed_dict.items():
        print('|{:20s}| {:6d}|'.format(word, count)) 
    print('----------------------------')

    # ## 3.2.4 POS Tagging
    random_5 = random.sample(range(1, len(df['reviewText'])), 5)
    random_5_df = df.iloc[random_5]
    # df.iloc[random_5]['tokenizedReview'] = df.iloc[random_5]['reviewText'].apply(tokenize, freq=False, unique=False)
    # df.iloc[random_5]['posTagged'] = df.iloc[random_5]['tokenizedReview'].apply(pos_tag)
    random_5_df['tokenizedReview'] = random_5_df['reviewText'].apply(tokenize, freq=False, unique=False)
    random_5_df['posTagged'] = random_5_df['tokenizedReview'].apply(pos_tag)
    display(random_5_df['posTagged'])
    print('==========================')
#end def


def top_n_words(df_series, stem=False, n=20):
    words_dict = dict()
    if stem:
        for review in df_series:
            tokenized_review = tokenize_and_stem(review, unique=False, freq=False)
            for token in tokenized_review:
                try:
                    words_dict[token] += 1
                except KeyError:
                    words_dict[token] = 1
            #end for
        #end for
    else:
        for review in df_series:
            tokenized_review = tokenize(review, unique=False, freq=False)
            for token in tokenized_review:
                try:
                    words_dict[token] += 1
                except KeyError:
                    words_dict[token] = 1
            #end for
        #end for
    #end if

    words_dict = OrderedDict(sorted(words_dict.items(), key=lambda t: t[1], reverse=True))
    
    i = 0
    returned_dict = OrderedDict()
    for word, count in words_dict.items():
        if i < n: returned_dict[word] = count
        else: break

        i += 1
    #end for

    return returned_dict
#end def


def tokenize(text, lower=True, remove_punc=True, stopwords=True, keep_emo=True, unique=True, freq=True, **kwargs):
    
    def _verify_emoticon(tmp_token, token):
        return (tmp_token + token) in EMOTICONS_TOKEN
    #end def

    def _emoticons_detection(tokenized_list):
        new_tokenized_list = list()
        n = len(tokenized_list)
        i = 0
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
                    else:
                        if _verify_emoticon(tmp_token, token):
                            new_tokenized_list.append(tmp_token + token)
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
        return new_tokenized_list
    #end def
    
    sentences = sent_tokenize(text)
    t = list()
    for s in sentences:
        tokenized = TreebankWordTokenizer().tokenize(s)
        tokenized = [SPECIAL_TOKEN[token] if SPECIAL_TOKEN.get(token, '') else token for token in tokenized]
        if lower:
            tokenized = [token.lower() for token in tokenized]
        if stopwords:
            tokenized = [token for token in tokenized if token not in STOPWORDS] 

        if keep_emo:
            t += _emoticons_detection(tokenized)
        else:
            t += tokenized

    if remove_punc:
        t = [token for token in t if token not in string.punctuation]

    if unique:
        t = set(t)

    if freq: return len(t)
    else: return t
    # return {token: t_list.count(token) for token in set(t_list)}
#end def


def tokenize_and_stem(text, unique=True, freq=True, **kwargs):
    tokenized = tokenize(text, freq=False, unique=False, **kwargs)
    stemmer = SnowballStemmer("english")

    if unique:
        t = {stemmer.stem(token) for token in tokenized}
    else:
        t = [stemmer.stem(token) for token in tokenized]

    if freq: return len(t)
    else: return t
#end def


def seg_sentences(text):
    # sentences = regex.split(r'[.?!]\s+|\.+\s+', text)
    sentences = sent_tokenize(text)
    return len([sentence for sentence in sentences if sentence])
#end def
       

if __name__ == '__main__': main()
