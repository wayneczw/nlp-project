import json
import os
import nltk
import pandas as pd
pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# %matplotlib inline
plt.style.use('seaborn-whitegrid')
plt.rcParams['savefig.facecolor']='white'
params = {'figure.figsize': (20,15),
            'axes.titlesize': 20}
plt.rcParams.update(params)
import plotly
import cufflinks as cf

def check_dir(directory):
    if not os.path.exists(directory):
        check_dir(os.path.dirname(directory))
        os.mkdir(directory)
        print("{:<6} Make directory: {}".format('[INFO]', directory))

data_path = './data/CellPhoneReview.json'
images_path = './images'
check_dir(images_path)


# load data
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))

data_df = pd.DataFrame.from_dict(data)
# data_df = data_df.sample(10000)

# # check data
# data_df.shape
# data_df.head()
# data_df.isnull().sum()
# data_df.nunique()

# refactor date
data_df['reviewDate'] = pd.to_datetime(data_df['unixReviewTime'], unit='s')
data_df = data_df.drop(columns = ['unixReviewTime', 'reviewTime'])

# # most frequent for each colomn
# for col in data_df.columns:
#     print('-' * 50)
#     print(col)
#     print(data_df[col].value_counts().head(10))

# # reviewText VS summary
# print(data_df[data_df['reviewText'] == ''].shape)
# print(data_df[data_df['summary'] == ''].shape)
# print(data_df[data_df['reviewText'] == data_df['summary']].shape)


# Question 3.2 -----------------------------------------------------------------
# A. Popular Products and Frequent Reviewers
data_df['asin'].value_counts().head(10).reset_index().rename(columns = {'index': 'productID', 'asin': 'reviewCount'})
data_df['reviewerID'].value_counts().head(10).reset_index().rename(columns = {'index': 'reviewerID', 'asin': 'reviewCount'})

# B. Sentence Segmentation
data_df['reviewSentenceTokenized'] = data_df['reviewText'].apply(lambda text: nltk.tokenize.sent_tokenize(text))
data_df['reviewSentenceCount'] = data_df['reviewSentenceTokenized'].apply(lambda text: len(text))

data_df[data_df['reviewText'].str.contains(':\)')][['reviewSentenceTokenized','reviewSentenceCount']]
data_df[data_df['reviewText'].str.contains('..')][['reviewText','reviewSentenceTokenized','reviewSentenceCount']]
data_df[['reviewText','reviewSentenceTokenized','reviewSentenceCount']].sort_values(['reviewSentenceCount'], ascending = False).head()

# plotting for number of sentences
sns.set(font_scale = 1.5)
fig = sns.countplot(data_df['reviewSentenceCount'].clip(0, 50), color = 'c')
for p in fig.patches:
    height = int(p.get_height())
    if height != 0:
        fig.text(p.get_x()+p.get_width()/2.,
                height + 140,
                height,
                ha="center", fontsize = 15)
title = 'Distribution of Number of Sentences for Each Review (Clipped)'
plt.title(title, loc = 'center', y=1.08, fontsize = 30)
fig.set_xlabel("Sentence Count (Clipped)")
fig.set_ylabel("Review Count")
plt.tight_layout()
saved_path = os.path.join(images_path, title.lower().replace(' ', '_'))
fig.get_figure().savefig(saved_path, dpi=200, bbox_inches="tight")



sns.set(font_scale = 2)
fig = sns.distplot(data_df['reviewSentenceCount'], kde = False,color = 'c')
title = 'Distribution of Number of Sentences for Each Review'
plt.title(title, loc = 'center', y=1.08, fontsize = 30)
fig.set_xlabel("Sentence Count")
fig.set_ylabel("Review Count")
plt.tight_layout()
saved_path = os.path.join(images_path, title.lower().replace(' ', '_'))
fig.get_figure().savefig(saved_path, dpi=200, bbox_inches="tight")


# C. Tokenization and Stemming.
data_df['reviewWordTokenized'] = data_df['reviewText'].apply(lambda text: nltk.tokenize.word_tokenize(text))
data_df['reviewWordCount'] = data_df['reviewWordTokenized'].apply(lambda text: len(text))
data_df[['reviewWordTokenized','reviewWordCount']].sort_values(['reviewWordCount'], ascending = False).head()
data_df[['reviewWordTokenized','reviewWordCount']].head()

sns.distplot(data_df['reviewWordCount'])
data_df['reviewWordCount'].iplot(kind = 'hist')
data_df['reviewWordCount'].clip(0,1000).iplot(kind = 'hist')


stemmer = nltk.stem.porter.PorterStemmer()
data_df['reviewWordStemmed'] = data_df['reviewWordTokenized'].apply(lambda text: [stemmer.stem(word) for word in text])
data_df['reviewWordCountStemmed'] = data_df['reviewWordStemmed'].apply(lambda text: len(text))
sns.distplot(data_df['reviewWordCountStemmed'])
data_df['reviewWordCountStemmed'].iplot(kind = 'hist')
data_df['reviewWordCountStemmed'].clip(0,1000).iplot(kind = 'hist')


sns.distplot(data_df['reviewWordCount'].clip(0,1000))
sns.distplot(data_df['reviewWordCountStemmed'].clip(0,1000))


# D. POS Tagging
sentence = 'I think this is one of the best cases I have ever bought and I like the fact that I can use either a stock battery or any number of extended batteries with it'
tokens = nltk.word_tokenize(sentence)
# nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(tokens)




#
# from nltk.tokenize import PunktSentenceTokenizer
# tokenizer = PunktSentenceTokenizer()
# tokenizer = PunktSentenceTokenizer()
#
# reviews_df = data_df[['reviewText']]
# reviews_df['reviewSentenceTokenized'] = reviews_df['reviewText'].apply(lambda text: nltk.tokenize.sent_tokenize(text))
# reviews_df['sentence_len'] = reviews_df['reviewSentenceTokenized'].apply(lambda list: [len(sent) for sent in list])
# reviews_df.head()
# reviews_df[reviews_df['sentence_len'].apply(lambda x: 2 in x)].head(10)
#
#
# sentence = 'These sim adapters work great 5:25 size SIM cards theses are a must :) Hello!!!! Hello?'
# stoppers = ['!', '.', '?']
# emojis = ['\:\)']
#
# stop_regex = '[{}]+'.format(''.join(['({})'.format(stopper) for stopper in stoppers + emojis]))
# sent_regex = '\\b[^{}]*'.format(''.join(stoppers))
#
#
# stoppers_regex
# from nltk.tokenize import RegexpTokenizer
# print(sent_regex + stop_regex)
# tokenizer = RegexpTokenizer(sent_regex + stop_regex)
#
# tokenizer = RegexpTokenizer('\\b(.+)(:\)|!)')
# tokenizer.tokenize(sentence)
# nltk.tokenize.sent_tokenize(sentence)
#
#
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
# class BulletPointLangVars(PunktLanguageVars):
#     sent_end_chars = ('.', '?', '!', ':\)')
#
# tokenizer = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
# tokenizer.tokenize(sentence)
