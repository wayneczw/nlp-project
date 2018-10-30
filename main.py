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
pd.set_option("display.max_rows", 999)

DEFAULT_DATA_FILE = "./data/sample_data.json"
IMAGES_DIRECTORY = './images'
REP_DIRECTORY = './rep_words'
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
    # make directory for representative words
    if not os.path.exists(REP_DIRECTORY):
        os.mkdir(REP_DIRECTORY)

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

    # get 5 random reviews to do sentence segmentation and display results
    reviews = df['reviewText']
    _seed = 43 # To give us an interesting result
    random_reviews = reviews.sample(5, random_state = _seed) 
    random_reviews = pd.DataFrame(random_reviews, columns = ['reviewText']).reset_index().drop(columns = ['index'])
    random_reviews['segmentedSentences'] = random_reviews['reviewText'].apply(segment_sent)
    print("5 Randomly selected reviews before and after sentence segmenetation:")
    print(random_reviews)

    ## 3.2.3 Tokenization and Stemming
    print_header('3.2.3 Tokenization and Stemming', 50)

    df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])
    df['tokens'] = df['tokenizedSentences'].apply(flatten)

    ### No Stemming
    print_header('No Stemming', char = '-')
    df['words'] = df['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
    df['words'] = df['words'].apply(lambda tokens: [token for token in tokens if is_word(token)])
    df['uniqueWords'] = df['words'].apply(set)
    df['wordCount'] = df['uniqueWords'].apply(len)

    # token = {normal_word, emoji, stopword, punctuation}
    # word = {normal_word, emoji}

    plot_bar(df['wordCount'], title = 'Distribution of Number of Words for Each Review Without Stemming',
            x_label = "Word Count", y_label = "Review Count", countplot = False)
    plot_bar(df['wordCount'].clip(0, 300), title = 'Distribution of Number of Words for Each Review Without Stemming (Clipped)',
            x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    words = flatten(df['words'])
    words_unique = flatten(df['uniqueWords'])

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
    df['uniqueStemmedWords'] = df['stemmedWords'].apply(set)
    df['stemmedWordCount'] = df['uniqueStemmedWords'].apply(len)

    plot_bar(df['stemmedWordCount'], \
            title = 'Distribution of Number of Words for Each Review With Stemming', \
            x_label = "Stemmed Word Count", y_label = "Review Count", countplot = False)
    plot_bar(df['stemmedWordCount'].clip(0, 300), \
            title = 'Distribution of Number of Words for Each Review With Stemming (Clipped)', \
            x_label = "Word Count (Clipped)", y_label = "Review Count", countplot = False)

    plot_bar_overlap(df, ['wordCount', 'stemmedWordCount'], \
            title = 'Distribution of Number of Words for Each Review', \
            x_label = "Word Count", y_label = "Review Count", countplot = False)

    plot_bar_overlap(df, ['wordCount', 'stemmedWordCount'], \
            title = 'Distribution of Number of Words for Each Review (Clipped)', \
            x_label = "Word Count", y_label = "Review Count", countplot = False)


    stemmed_words = flatten(df['stemmedWords'])
    stemmed_words_unique = flatten(df['uniqueStemmedWords'])

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

    # 3.3 Development of a Noun Phrase Summarizer
    print_header('3.3 Development of a Noun Phrase Summarizer', 50)

    df['posTagged'] = df['tokenizedSentences'].apply(lambda tokenizedSentences: [pos_tag(sentence) for sentence in tokenizedSentences])
    df['nounPhrases'] = df['posTagged'].apply(lambda posTagged: [np.lower() for np in flatten([extract_NP(tag) for tag in posTagged])])
    df[['reviewText', 'posTagged', 'nounPhrases']].head()


    # Including single noun phrases
    print_header('Including single noun phrases', char = '-')
    noun_phrases = pd.DataFrame.from_dict(Counter(flatten(df['nounPhrases'])), orient='index').\
                    reset_index().rename(columns = {'index': 'Noun Phrase', 0: 'Count'}).\
                    sort_values(['Count'], ascending = False)
    top_20_noun_phrases = noun_phrases.head(20).reset_index().drop(columns = ['index'])

    print_header('Top 20 Noun Phrases Including Single Noun Phrases', char = '-')
    print(top_20_noun_phrases)
    #      Noun Phrase   Count
    # 0   i             561124
    # 1   it            541931
    # 2   you           161508
    # 3   they          57945
    # 4   me            44179
    # 5   the phone     38242
    # 6   them          31139
    # 7   the case      27750
    # 8   this case     27186
    # 9   my phone      24061
    # 10  we            12584
    # 11  the price     11420
    # 12  this product  10920
    # 13  she           10162
    # 14  your phone    10012
    # 15  the screen    9802
    # 16  the battery   8408
    # 17  my iphone     8359
    # 18  something     8165
    # 19  a bit         8038

    products = df['asin'].value_counts().head(3).index
    products_np_top1 = df[df['asin']== products[0]]
    products_np_top2 = df[df['asin']== products[1]]
    products_np_top3 = df[df['asin']== products[2]]
  
    print_representative_np(products_np_top1, product=products[0], n=30)
    print_representative_np(products_np_top2, product=products[1], n=30)
    print_representative_np(products_np_top3, product=products[2], n=30)


    # noun_phrases_count_map = noun_phrases.set_index('Noun Phrase').to_dict()['Count']

    # products = df['asin'].value_counts().head(3).index
    # products_np = df[df['asin'].isin(products)][['asin', 'nounPhrases']].groupby(['asin']).sum().reset_index()
    # products_np['NPCounts'] = products_np['nounPhrases'].apply(lambda noun_phrases: Counter(noun_phrases).most_common())

    # # Absolute Count
    # products_np['abosoluteTop10'] = products_np['NPCounts'].apply(lambda NPCounts: NPCounts[:10])

    # # Relative counts
    # products_np['relativeNPFrequency'] = products_np['NPCounts'].apply(lambda NPCounts: [(NPCount[0], NPCount[1]/noun_phrases_count_map[NPCount[0]]) for NPCount in NPCounts])
    # products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency.sort(key=lambda elem: elem[1], reverse=True))
    # products_np['relativeTop10'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency[:10])
    # products_np['relative1'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: [word for word in relativeNPFrequency if word[1] == 1])

    # product_representative_NPs = products_np[['asin', 'abosoluteTop10', 'relativeTop10', 'relative1']].rename(columns = {'asin': 'productID', 'abosoluteTop10': '10 Representative Noun Phrases (Abosolute)', 'relativeTop10': '10 Representative Noun Phrases (Relative)', 'relative1': 'Representative Noun Phrases (Unique)'})

    # print_header('10 Representative Noun Phrases for 3 Popular Products (Including single noun phrases)', char = '-')

    # print_header('Abosolute', char = '-')
    # print(pd.DataFrame({row['asin']: row['abosoluteTop10'] for index, row in products_np.iterrows()}))

    # print_header('Relative', char = '-')
    # print(pd.DataFrame({row['asin']: row['relativeTop10'] for index, row in products_np.iterrows()}))

    # print_header('Unique', char = '-')
    # unique_count_max = (products_np['relative1'].apply(len)).max()
    # products_np['relative1'] = products_np['relative1'].apply(lambda relative1: relative1 + [''] * (unique_count_max - len(relative1)))
    # print(pd.DataFrame({row['asin']: row['relative1'] for index, row in products_np.iterrows()}))


    # #     productID  \
    # # 0  B0042FV2SI
    # # 1  B005SUHPO6
    # # 2  B008OHNZI0
    # #
    # #                                                                                                                          10 Representative Noun Phrases (Abosolute)  \
    # # 0  [(it, 1127), (i, 1046), (you, 245), (they, 162), (them, 90), (my phone, 76), (the screen, 73), (me, 53), (this product, 37), (the phone, 35)]
    # # 1  [(it, 1956), (i, 1884), (you, 448), (this case, 228), (the case, 187), (the phone, 184), (they, 162), (me, 153), (my phone, 148), (he, 128)]
    # # 2  [(i, 1728), (it, 1319), (you, 464), (they, 251), (me, 126), (the screen, 125), (them, 112), (the screen protector, 73), (the protector, 58), (this product, 56)]
    # #
    # #                                                                                                                                                                                                                                                                                         10 Representative Noun Phrases (Relative)
    # # 0  [(the matte finishing, 1.0), (a deeper scratch on the protector, 1.0), (didnt stay, 1.0), (bubble free surface, 1.0), (the 30th amazing, 1.0), (lights indoors, 1.0), (using screen covers by generic for all long time, 1.0), (that well on my iphone, 1.0), (best investment for any smartphone, 1.0), (sprint 's i4s, 1.0)]
    # # 1  [(otterbox defender series hybrid case, 1.0), (its quite annoying, 1.0), (charging port keeps, 1.0), (an otterbox for my ipad, 1.0), (the line of the case, 1.0), (love thesei, 1.0), (best case for the iphone, 1.0), (is a fake, 1.0), (this casse, 1.0), (might work for others, 1.0)]
    # # 2  [(gos:, 1.0), (ta:, 1.0), (the scotch tape method, 1.0), (the high definition, 1.0), (tab number, 1.0), (the tech armor hd clear screen protector, 1.0), (the lint lifter, 1.0), (accomplishment, 1.0), (perfect screen protectors, 1.0), (the home button side, 1.0)]

    # # Excluding single noun phrases
    # print_header('Excluding single noun phrases', char = '-')

    # df['nounPhrasesExcludeSingle'] = df['nounPhrases'].apply(lambda noun_phrases: [noun_phrase for noun_phrase in noun_phrases if len(noun_phrase.split()) > 1])
    # noun_phrases = pd.DataFrame.from_dict(Counter(flatten(df['nounPhrasesExcludeSingle'])), orient='index').\
    #                 reset_index().rename(columns = {'index': 'Noun Phrase', 0: 'Count'}).\
    #                 sort_values(['Count'], ascending = False)
    # top_20_noun_phrases = noun_phrases.head(20).reset_index().drop(columns = ['index'])

    # print_header('Top 20 Noun Phrases Excluding Single Noun Phrases', char = '-')
    # print(top_20_noun_phrases)
    # #      Noun Phrase  Count
    # # 0   the phone     38242
    # # 1   the case      27750
    # # 2   this case     27186
    # # 3   my phone      24061
    # # 4   the price     11420
    # # 5   this product  10920
    # # 6   your phone    10012
    # # 7   the screen    9802
    # # 8   the battery   8408
    # # 9   my iphone     8359
    # # 10  a bit         8038
    # # 11  the iphone    6440
    # # 12  this phone    5796
    # # 13  a lot         5680
    # # 14  the charger   5346
    # # 15  the way       5158
    # # 16  the device    4873
    # # 17  the time      4468
    # # 18  the product   4451
    # # 19  a case        4145

    # noun_phrases_count_map = noun_phrases.set_index('Noun Phrase').to_dict()['Count']
    # products = df['asin'].value_counts().head(3).index
    # products_np = df[df['asin'].isin(products)][['asin', 'nounPhrasesExcludeSingle']].groupby(['asin']).sum().reset_index()
    # products_np['NPCountsExcludeSingle'] = products_np['nounPhrasesExcludeSingle'].apply(lambda noun_phrases: Counter(noun_phrases).most_common())

    # # Absolute Count
    # products_np['abosoluteTop10ExcludeSingle'] = products_np['NPCountsExcludeSingle'].apply(lambda NPCounts: NPCounts[:10])

    # # Relative counts
    # products_np['relativeNPFrequencyExcludeSingle'] = products_np['NPCountsExcludeSingle'].apply(lambda NPCounts: [(NPCount[0], NPCount[1]/noun_phrases_count_map[NPCount[0]]) for NPCount in NPCounts])
    # products_np['relativeNPFrequencyExcludeSingle'].apply(lambda relativeNPFrequency: relativeNPFrequency.sort(key=lambda elem: elem[1], reverse=True))
    # products_np['relativeTop10ExcludeSingle'] = products_np['relativeNPFrequencyExcludeSingle'].apply(lambda relativeNPFrequency: relativeNPFrequency[:10])
    # products_np['relative1ExcludeSingle'] = products_np['relativeNPFrequencyExcludeSingle'].apply(lambda relativeNPFrequency: [word for word in relativeNPFrequency if word[1] == 1])

    # product_representative_NPs = products_np[['asin', 'abosoluteTop10ExcludeSingle', 'relativeTop10ExcludeSingle', 'relative1ExcludeSingle']]\
    #     .rename(columns = {'asin': 'productID', 'abosoluteTop10ExcludeSingle': '10 Representative Noun Phrases (Abosolute)', 'relativeTop10ExcludeSingle': '10 Representative Noun Phrases (Relative)', 'relative1ExcludeSingle': 'Representative Noun Phrases (Unique)'})

    # print_header('10 Representative Noun Phrases for 3 Popular Products (Excluding single noun phrases)', char = '-')

    # print_header('Abosolute', char = '-')
    # print(pd.DataFrame({row['asin']: row['abosoluteTop10ExcludeSingle'] for index, row in products_np.iterrows()}))

    # #     B0042FV2SI                   B005SUHPO6              B008OHNZI0
    # # 0  (my phone, 76)               (this case, 228)        (the screen, 125)
    # # 1  (the screen, 73)             (the case, 187)         (the screen protector, 73)
    # # 2  (this product, 37)           (the phone, 184)        (the protector, 58)
    # # 3  (the phone, 35)              (my phone, 148)         (this product, 56)
    # # 4  (the price, 30)              (your phone, 72)        (tech armor, 53)
    # # 5  (this screen protector, 28)  (my iphone, 55)         (the phone, 52)
    # # 6  (the screen protector, 24)   (the iphone, 48)        (my iphone, 50)
    # # 7  (my iphone, 24)              (this product, 47)      (my phone, 47)
    # # 8  (the matte finish, 23)       (the price, 42)         (no bubbles, 45)
    # # 9  (the product, 23)            (the color, 37)         (the price, 44)

    # print_header('Relative', char = '-')
    # print(pd.DataFrame({row['asin']: row['relativeTop10ExcludeSingle'] for index, row in products_np.iterrows()}))

    # print_header('Unique', char = '-')
    # unique_count_max = (products_np['relative1ExcludeSingle'].apply(len)).max()
    # products_np['relative1ExcludeSingle'] = products_np['relative1ExcludeSingle'].apply(lambda relative1: relative1 + [''] * (unique_count_max - len(relative1)))
    # print(pd.DataFrame({row['asin']: row['relative1ExcludeSingle'] for index, row in products_np.iterrows()}))

    # # 0  [(the matte finishing, 1.0), (didnt stay, 1.0), (a deeper scratch on the protector, 1.0), (bubble free surface, 1.0), (the 30th amazing, 1.0), (lights indoors, 1.0), (using screen covers by generic for all long time, 1.0), (a clean microfiber cloth/eyeglass cloth, 1.0), (that well on my iphone, 1.0), (best investment for any smartphone, 1.0)]
    # # 1  [(otterbox defender series hybrid case, 1.0), (its quite annoying, 1.0), (charging port keeps, 1.0), (an otterbox for my ipad, 1.0), (the line of the case, 1.0), (love thesei, 1.0), (best case for the iphone, 1.0), (is a fake, 1.0), (this casse, 1.0), (might work for others, 1.0)]
    # # 2  [(the scotch tape method, 1.0), (the high definition, 1.0), (tab number, 1.0), (the tech armor hd clear screen protector, 1.0), (the home button side, 1.0), (the lint lifter, 1.0), (perfect screen protectors, 1.0), (hate fingerprints, 1.0), (retinashield screen protector, 1.0), (vs personal phone, 1.0)]


    random_5_reviews = df[['reviewText', 'posTagged', 'nounPhrases']].sample(5, random_state = seed)
    random_5_reviews['nounPhrasesLen'] = random_5_reviews['nounPhrases'].apply(len)

    print_header('Noun Phrase Detector Evaluation for  Random 5 Reviews', char = '-')
    print(random_5_reviews)

    #                         reviewText  \
    # 178468  This case took over a month to reach my resident.  This case was suppose to be for a Galaxy Note 3; the case did not work for my phone.  The part of the case that holds the phone in the case was missing.  But even if the part was there this case would not have worked because the bottom portion where you answer/disconnect the call on the case is much higher up than the answer/disconnect on my phone.  It was only $3.83 but it wasn't worth that amount because it didn't work.  I threw it in the trash.
    # 84136   A good sturdy TPU case with perfect cutouts. No buttons are hard to push. The product has a tight fit and definitely won't slide off unless you try hard to pry it off. Awesome case that I used on my Nexus. Now that the wife has it, she loves the case too. Great deal.
    # 189736  This is an impressively sturdy case for the Samsung galaxy S5 at an incredible price.  The case fits well, does a great job of protecting the phone, and the kickstand is a pleasant and unexpected feature.
    # 105362  This is a great stylus.  I have had the rubber tip ones and they always tear and aren't as durable.  The mesh on this is very durable and sensitive. Works perfectly.
    # 21769   The item was as described. I am completely satisfied. The protector does what it is supposed to. I would reccommend this product to anyone.
    #
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            posTagged  \
    # 178468  [[(This, DT), (case, NN), (took, VBD), (over, RP), (a, DT), (month, NN), (to, TO), (reach, VB), (my, PRP$), (resident, NN), (., .)], [(This, DT), (case, NN), (was, VBD), (suppose, JJ), (to, TO), (be, VB), (for, IN), (a, DT), (Galaxy, NNP), (Note, NNP), (3, CD), (;, :), (the, DT), (case, NN), (did, VBD), (not, RB), (work, NN), (for, IN), (my, PRP$), (phone, NN), (., .)], [(The, DT), (part, NN), (of, IN), (the, DT), (case, NN), (that, WDT), (holds, VBZ), (the, DT), (phone, NN), (in, IN), (the, DT), (case, NN), (was, VBD), (missing, VBG), (., .)], [(But, CC), (even, RB), (if, IN), (the, DT), (part, NN), (was, VBD), (there, RB), (this, DT), (case, NN), (would, MD), (not, RB), (have, VB), (worked, VBN), (because, IN), (the, DT), (bottom, JJ), (portion, NN), (where, WRB), (you, PRP), (answer/disconnect, VBP), (the, DT), (call, NN), (on, IN), (the, DT), (case, NN), (is, VBZ), (much, RB), (higher, JJR), (up, RB), (than, IN), (the, DT), (answer/disconnect, NN), (on, IN), (my, PRP$), (phone, NN), (., .)], [(It, PRP), (was, VBD), (only, RB), ($3.83, JJ), (but, CC), (it, PRP), (was, VBD), (not, RB), (worth, JJ), (that, IN), (amount, NN), (because, IN), (it, PRP), (did, VBD), (not, RB), (work, NN), (., .)], [(I, PRP), (threw, VBD), (it, PRP), (in, IN), (the, DT), (trash, NN), (., .)]]
    # 84136   [[(A, DT), (good, JJ), (sturdy, NN), (TPU, NNP), (case, NN), (with, IN), (perfect, JJ), (cutouts, NNS), (., .)], [(No, DT), (buttons, NNS), (are, VBP), (hard, JJ), (to, TO), (push, VB), (., .)], [(The, DT), (product, NN), (has, VBZ), (a, DT), (tight, JJ), (fit, NN), (and, CC), (definitely, RB), (wo, MD), (not, RB), (slide, VB), (off, RP), (unless, IN), (you, PRP), (try, VBP), (hard, JJ), (to, TO), (pry, VB), (it, PRP), (off, RP), (., .)], [(Awesome, NNP), (case, NN), (that, WDT), (I, PRP), (used, VBD), (on, IN), (my, PRP$), (Nexus, NNP), (., .)], [(Now, RB), (that, IN), (the, DT), (wife, NN), (has, VBZ), (it, PRP), (,, ,), (she, PRP), (loves, VBZ), (the, DT), (case, NN), (too, RB), (., .)], [(Great, JJ), (deal, NN), (., .)]]
    # 189736  [[(This, DT), (is, VBZ), (an, DT), (impressively, RB), (sturdy, JJ), (case, NN), (for, IN), (the, DT), (Samsung, NNP), (galaxy, NN), (S5, NNP), (at, IN), (an, DT), (incredible, JJ), (price, NN), (., .)], [(The, DT), (case, NN), (fits, VBZ), (well, RB), (,, ,), (does, VBZ), (a, DT), (great, JJ), (job, NN), (of, IN), (protecting, VBG), (the, DT), (phone, NN), (,, ,), (and, CC), (the, DT), (kickstand, NN), (is, VBZ), (a, DT), (pleasant, JJ), (and, CC), (unexpected, JJ), (feature, NN), (., .)]]
    # 105362  [[(This, DT), (is, VBZ), (a, DT), (great, JJ), (stylus, NN), (., .)], [(I, PRP), (have, VBP), (had, VBD), (the, DT), (rubber, NN), (tip, NN), (ones, NNS), (and, CC), (they, PRP), (always, RB), (tear, VBP), (and, CC), (are, VBP), (not, RB), (as, IN), (durable, JJ), (., .)], [(The, DT), (mesh, NN), (on, IN), (this, DT), (is, VBZ), (very, RB), (durable, JJ), (and, CC), (sensitive, JJ), (., .)], [(Works, VBZ), (perfectly, RB), (., .)]]
    # 21769   [[(The, DT), (item, NN), (was, VBD), (as, IN), (described, NN), (., .)], [(I, PRP), (am, VBP), (completely, RB), (satisfied, JJ), (., .)], [(The, DT), (protector, NN), (does, VBZ), (what, WP), (it, PRP), (is, VBZ), (supposed, VBN), (to, TO), (., .)], [(I, PRP), (would, MD), (reccommend, VB), (this, DT), (product, NN), (to, TO), (anyone, NN), (., .)]]
    #
    #                                                                                                                                                                                                                                                                                      nounPhrases
    # 178468  [this case, a month, my resident, this case, a galaxy note, the case, work for my phone, the part of the case, the phone in the case, the part, this case, the bottom portion, you, the call on the case, the answer/disconnect on my phone, it, it, amount, it, work, i, it, the trash]
    # 84136   [a good sturdy tpu case with perfect cutouts, no buttons, the product, a tight fit, you, it, awesome case, i, my nexus, the wife, it, she, the case, great deal]
    # 189736  [an impressively sturdy case for the samsung galaxy s5 at an incredible price, the case, a great job, the phone, the kickstand, a pleasant and unexpected feature]
    # 105362  [a great stylus, i, the rubber tip ones, they, the mesh]
    # 21769   [the item, described, i, the protector, it, i, this product, anyone]

    # Manually
    # 178468  [this case, a month, my resident, this case, a galaxy note 3, the case, my phone, the part of the case, the phone in the case, the part, this case, the bottom portion, you, the call on the case, the answer/disconnect on my phone, it, it, that amount, it, i, it, the trash]
    # 84136   [a good sturdy tpu case with perfect cutouts, no buttons, the product, a tight fit, you, it, awesome case, i, my nexus, the wife, it, she, the case, great deal]
    # 189736  [an impressively sturdy case for the samsung galaxy s5 at an incredible price, the case, a great job, the phone, the kickstand, a pleasant and unexpected feature]
    # 105362  [a great stylus, i, the rubber tip ones, they, the mesh]
    # 21769   [the item, i, the protector, it, i, this product, anyone]

    # @TODO
    # Recall + Precision



    # 3.4. Sentiment Word Detection
    print(str(datetime.datetime.now()).split('.')[0] + ': Start processing sentence segmentation')

    # Without Stemming and Without Negation
    sentiment_score(df, "./rep_words/ns_nn.csv")

    # With Stemming and Without Negation
    sentiment_score(df, "./rep_words/s_nn.csv", stemmer=stemmer)

    # Without Stemming and With Negation
    sentiment_score(df, "./rep_words/ns_n.csv", convert_neg=True)

    # With Stemming and With Negation
    sentiment_score(df, "./rep_words/s_n.csv", stemmer=stemmer, convert_neg=True)

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
#end class


def print_representative_np(df, product, n=50):
    def _identity_tokenizer(text):
        return text

    tfidf = TfidfVectorizer(tokenizer=_identity_tokenizer, stop_words='english', lowercase=False)    
    try:
        result = tfidf.fit_transform(df['nounPhrases'])
    except Exception as e:
        df['posTagged'] = df['tokenizedSentences'].apply(lambda tokenizedSentences: [pos_tag(sentence) for sentence in tokenizedSentences])
        df['nounPhrases'] = df['posTagged'].apply(lambda posTagged: [np.lower() for np in flatten([extract_NP(tag) for tag in posTagged])])
        result = tfidf.fit_transform(df['nounPhrases'])

    scores = zip(tfidf.get_feature_names(),
                 np.asarray(result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print('='*30 + product + '='*30)
    for item in sorted_scores[:n]:
        print("{0:50} Score: {1}".format(item[0], item[1]))
    print()
    print()
#end def


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

def _convert_neg(tokens,window_size = 4):
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
            i_window_limit = i + 4
            while (i < n and i < i_window_limit) :
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
#end def


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


def sentiment_score(df, path="./rep_words/rep_words.csv", stemmer=None, convert_neg=False):
    df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence, stemmer = stemmer, remove_punc = True, remove_stopwords = True, remove_emoji = False, convert_neg = convert_neg) for sentence in sentences])
    df['tokens'] = df['tokenizedSentences'].apply(flatten)
    df['words'] = df['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
    df['words'] = df['words'].apply(lambda tokens: [token for token in tokens if is_word(token)])

    it_score_dict = it_score(df)
    probabilistic_score_dict = probabilistic_score(df)
    word_score_dict = dict()
    for word in probabilistic_score_dict.keys():
        word_score_dict[word] = (probabilistic_score_dict[word] + it_score_dict[word]) / 2

    orderd_word_score_dict = OrderedDict(sorted(word_score_dict.items(), key=lambda t: t[1], reverse=True))
    pd.DataFrame.from_dict(orderd_word_score_dict, orient="index").to_csv(path)
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
