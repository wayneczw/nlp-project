import json
import pandas as pd
from pandas.io.json import json_normalize
import nltk
import matplotlib.pyplot as plt
import os
from nltk.tokenize import TweetTokenizer
# create stop_words list
from nltk.corpus import stopwords
# Load stop words
stop_words = stopwords.words('english')
tknzr = TweetTokenizer()
stemmer = nltk.stem.porter.PorterStemmer()

# def tokenize(text, tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True))
def tokenize(text, tokenizer = tknzr):
    # Use TweetTokenizer to recognize text emoticon such as :) :-)
    return [ tokenizer.tokenize(sentence) for sentence in sent_tokenize(text) ]

df = pd.read_json('./data/CellPhoneReview.json', orient='columns', lines=True)
# top-10 products that attract the most number of reviews
top_10_products = df.groupby('asin').size().nlargest(10).reset_index(name='no. of reviews')
print(top_10_products)
# top-10 reviewers who have contributed most number of reviews
top_10_reviewers = df.groupby('reviewerID').size().nlargest(10).reset_index(name='no. of reviews')
print(top_10_reviewers)

# Sentence Segmentation
df['reviewTextSentenceSegmented'] = df['reviewText'].map(lambda text: nltk.tokenize.sent_tokenize(text))
# sentence length
df['reviewSentenceLength'] = df['reviewTextSentenceSegmented'].map(lambda segmentedSentence: len(segmentedSentence))
df_review_sentence = df.groupby('reviewSentenceLength').size().reset_index(name='reviewTextCounts')

if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
plt.figure(1)
plt.plot('reviewSentenceLength','reviewTextCounts','.-',data=df_review_sentence)
plt.xlabel('length of a review in number of sentences')
plt.ylabel('number of reviews of each length')
plt.savefig('./figures/review counts vs sentence length.png')
plt.show()

'''
Randomly sample 5 reviews (including both short reviews and long reviews) and verify whether the
sentence segmentation function/tool detects the sentence boundaries correctly. Discuss your results.
'''
# short_sentence_length <= 25%
short_sentence_length = df['reviewSentenceLength'].quantile(0.25)

# long_sentence_length <= 75%
long_sentence_length = df['reviewSentenceLength'].quantile(0.75)

sample_5_short = df[df['reviewSentenceLength']<=short_sentence_length].sample(n=5)
sample_5_long = df[df['reviewSentenceLength']<=long_sentence_length].sample(n=5)
sample_5_short.index = range(len(sample_5_short))
sample_5_long.index = range(len(sample_5_long))

# full-stop without space
# Wrong: should be 3 sentences
print(sample_5_short['reviewTextSentenceSegmented'][0])
## ['LOOKS CUTE IN PICTURE,TERRIBLE IN REAL LIFE, WAS GOING TO GIVE AS GIFT.', '& NOW CANNOTEXTREMEMLY DISSAPOINTED.VERY VERY CHEAP LOOKING']

# Correct: 2 sentences
print(sample_5_short['reviewTextSentenceSegmented'][1])
## ["It was everything I'd hoped...and it matches the color of my phone...a bonus.", "It's nice to have what I actually expected."]

# Correct: 2 sentences
print(sample_5_short['reviewTextSentenceSegmented'][2])
##['These screens are great quality and are easy to replace, just be careful with all the different ribbons.', 'I would buy these again.']

# ".." should be recognized as sentence boundary in some cases
# Wrong: should be 7 sentences
print(sample_5_short['reviewTextSentenceSegmented'][3])
##['The size grow on you within a few days..All other phones feel like toys there on..If you love Tech there is always something new to try on this phone..It will make you forget your Tablets if you own one..One of the only phones on the market that still turn heads..check out my Video reviews on my YouTube Chanel...First look https://www.youtube.com/watch?v=JQZjn4U4lgY12 hrs later  https://www.youtube.com/watch?v=XFDZxENFqF8']

# "() should be considered in one sentence, they are a pair"
# ! without space
# Wrong: should be 3 sentences
print(sample_5_short['reviewTextSentenceSegmented'][4])
## ['Great little charger!My original was partially chewed by my puppy (thumbs down!', ').It works quickly and just as well as the original, plus the price is right!']


# Tokenization
df['reviewTextWordTokenized'] = df['reviewText'].map(lambda text: tknzr.tokenize(text.lower()))

# Remove Stop Words
df['reviewTextWordTokenized'] = df['reviewTextWordTokenized'].map(lambda tokens: [token for token in tokens if token not in stop_words])

tokens_concat=[]
df['reviewTextWordTokenized'].map(lambda x: tokens_concat.extend(x))
word_dist = nltk.FreqDist(tokens_concat)
word_freq = pd.DataFrame.from_dict(word_dist, orient='index')
word_freq = word_freq.reset_index()
word_freq.columns = ['token','counts']
word_freq = word_freq.sort_values(by='counts', ascending=False)
word_freq.to_csv('token_count.csv',index=False)

token_list = word_freq['token'].tolist()

token_doc_count =  {}
#initialization
for token in token_list:
    token_doc_count[token] = 0

list_review_token_set = []
for index,row in df.iterrows():
    list_review_token_set.append(row['reviewTextWordTokenized'])

for review_token_set in list_review_token_set:
    for token in review_token_set:
        token_doc_count[token] += 1

df_token_doc_count = pd.DataFrame.from_dict(token_doc_count, orient='index')
df_token_doc_count = df_token_doc_count.reset_index()
df_token_doc_count.columns = ['token','no. of reviews']
df_token_doc_count = df_token_doc_count.sort_values(by='no. of reviews', ascending=False)
df_token_doc_count.to_csv('df_token_doc_count.csv',index=False)

appended_stopwords = ['.',',','(',')','-','use','would','get','also','time',':','...','/','i\'m','i\'ve',';','&']
stop_words.extend(appended_stopwords)

# Remove Stop Words
df['reviewTextWordTokenized'] = df['reviewTextWordTokenized'].map(lambda tokens: [token for token in tokens if token not in stop_words])

word_freq = word_freq[~word_freq['token'].isin(stop_words)]
print(word_freq.head(20))

# stemming
df['reviewTextWordStemmed'] = df['reviewTextWordTokenized'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

stemmed_concat=[]
df['reviewTextWordStemmed'].map(lambda x: stemmed_concat.extend(x))
stemmed_dist = nltk.FreqDist(stemmed_concat)
stemmed_freq = pd.DataFrame.from_dict(stemmed_dist, orient='index')
stemmed_freq = stemmed_freq.reset_index()
stemmed_freq.columns = ['stemmed','counts']
stemmed_freq = stemmed_freq.sort_values(by='counts', ascending=False)
stemmed_freq.head(20)
