import os
import pandas as pd
import numpy as np
from nltk.corpus import wordnet

IACcorpus_dir = 'IACcorpus'

iac_dict = {}
for file in os.listdir(IACcorpus_dir):
    if file != '.DS_Store':
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

aspects = []
implicits = []
for k, v in iac_dict.items():
    for implicit in v:
        aspects.append(k)
        implicits.append(implicit)
iac_df = pd.DataFrame(data = {'aspect': aspects, 'implicit': implicits})

sent_df = pd.read_csv('rep_words/ns_nn.csv')
sent_df.columns = ['implicit', 'polarity_intense']

iac_polarity_df = iac_df.merge(sent_df, on = 'implicit', how = 'inner')

iac_polarity_df.to_csv('implicit_aspect_polarity.csv', index = False)
