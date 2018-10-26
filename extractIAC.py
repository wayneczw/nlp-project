import os
import pandas as pd
import numpy as np
from nltk.corpus import wordnet

IACcorpus_dir = 'IACcorpus'

from senticnet.senticnet import SenticNet
sn = SenticNet()

def get_polarity_value(word):
    try:
        return sn.polarity_value(word)
    except:
        return np.nan

def get_polarity_intense(word):
    try:
        return sn.polarity_intense(word)
    except:
        return np.nan


iac_dict = {}
for file in os.listdir(IACcorpus_dir):
    corpus_path = os.path.join(IACcorpus_dir, file)
    with open(corpus_path) as f:
        lines = f.readlines()

    for line in lines:
        if ']##' in line:
            iacs = line.split('##')[0].split(', ')
            for iac in iacs:
                explicit = iac.split('[')[-1].replace(']', '').lower()
                implicit = iac.split('[')[0].lower()
                if explicit not in iac_dict.keys():
                    iac_dict[explicit] = {implicit}
                else:
                    iac_dict[explicit].add(implicit)

for key, value in iac_dict.items():

    iac_set = value

    for word in value:
        syns = {syn.lemmas()[0].name().lower() for syn in wordnet.synsets(word)}
        ants = {syn.lemmas()[0].antonyms()[0].name().lower() for syn in wordnet.synsets(word) if syn.lemmas()[0].antonyms()}
        iac_set = iac_set.union(syns).union(ants)

    iac_dict[key] = iac_set

[len(implicit) for implicit in list(iac_dict.values())]
sum([len(implicit) for implicit in list(iac_dict.values())])
# iac_dict = {k: list(v) for k, v in iac_dict.items()}
iac_dict.keys()
iac_dict

aspects = []
implicits = []
for k, v in iac_dict.items():
    for implicit in v:
        aspects.append(k)
        implicits.append(implicit)
iac_df = pd.DataFrame(data = {'aspect': aspects, 'implicit': implicits})

iac_df['polarity_value'] = iac_df['implicit'].apply(get_polarity_value)
iac_df['polarity_intense'] = iac_df['implicit'].apply(get_polarity_intense)
iac_df = iac_df[~pd.isna(iac_df['polarity_value'])]

iac_df.isnull().sum()
iac_df.shape

iac_df[iac_df['implicit'] == 'expensive']

# iac_df.to_csv('implicit_aspect_corpus.csv', index = False)
