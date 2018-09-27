import numpy as np
import json
import os
import nltk
import pandas as pd
from collections import Counter
pd.set_option('display.max_colwidth', -1)

data_path = './data/CellPhoneReview.json'
data_path = './data/sample_data.json'

flatten = lambda l: [item for sublist in l for item in sublist]

# load data
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['unixReviewTime', 'reviewTime'])


df['SegementedSentences'] = data_df['reviewText'].apply(nltk.tokenize.sent_tokenize)
df['TokenizedWordBySentence'] = df['SegementedSentences'].apply(lambda sentences: [nltk.tokenize.word_tokenize(sentence) for sentence in sentences])
data_df['reviewText']
sentences_df = pd.DataFrame(pd.Series(flatten(df['SegementedSentences'])), columns = ['OriginalSentence']).reset_index().drop(columns = ['index'])
sentences_df['TokenizedSentence'] = pd.Series(flatten(df['TokenizedWordBySentence']))
sentences_df['PosTagged'] = sentences_df['TokenizedSentence'].apply(nltk.pos_tag)
sentences_df['PosTagged']
