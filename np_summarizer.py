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
    # Append whitespace after punctuations, except .com
    for p in '.?!':
        regex = r'(?<=[^\d]+)\{}(?=[^ \W])(?!com)'.format(p)
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



def extract_NP(posTagged):
    grammar = r"""

        ADJ:
            {<RB.*>? <JJ.* | VBG>}

        ADJLIST:
            {<ADJ> (<CC>? <,>? <ADJ>)*}

        NOUN:
            {<DT|PRP\$>? (<NN.*> <POS>)? <ADJLIST>? <NN.*>+}

        NP:
            {<NOUN> (<IN><NOUN>)*}
            {<PRP>}

        """
    chunker = RegexpParser(grammar)
    ne = []
    chunk = chunker.parse(posTagged)
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.append(' '.join([child[0] for child in tree.leaves()]))
    return ne

# <JJ.*> —> JJ | JJR | JJS
# <NN.*> —> NN | NNS | NNP | NNPS

# {<JJ.*> (<CC>? <,>? <JJ.*>)*}  adj followed by optional ',' / 'conjunction' and adj

# {<DT|PRP\$> <VBG> <NN.*>+} eg. a sle
# {<DT|PRP\$> <NN.*> <POS> <ADJ>* <NN.*>+}
# {<DT|PRP\$>? <ADJ>* <NN.*>+ }
# {<PRP>}

# extract_NP(pos_tag(tokenize("I have the nicest, cheap and good phone case in the big shelf at home")))
# pos_tag(tokenize("It has a really weird and annoying sound from the back"))
# extract_NP(pos_tag(tokenize("on that person's face")))
# extract_NP(pos_tag(tokenize("It has a most weird and annoying sound from the back")))
#
# extract_NP(pos_tag(tokenize("I have a always dying and good phone case in the big shelf at home")))
# extract_NP(pos_tag(tokenize("I like my friend's pretty phone")))


data_path = "./data/CellPhoneReview.json"
data_path = "./data/sample_data.json"
data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame.from_dict(data)
df = df.drop(columns = ['overall', 'reviewTime', 'summary', 'unixReviewTime'])




df['sentences'] = df['reviewText'].apply(segment_sent)
df['tokenizedSentences'] = df['sentences'].apply(lambda sentences: [tokenize(sentence) for sentence in sentences])
df['posTagged'] = df['tokenizedSentences'].apply(lambda tokenizedSentences: [pos_tag(sentence) for sentence in tokenizedSentences])
df['nounPhrases'] = df['posTagged'].apply(lambda posTagged: [np.lower() for np in flatten([extract_NP(tag) for tag in posTagged])])
df[['reviewText', 'posTagged', 'nounPhrases']].head()
df.head()
seed = 42
random_5_reviews = df[['reviewText', 'posTagged', 'nounPhrases']].sample(5, random_state = seed)
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
# 178468  [this case, a month, my resident, this case, a galaxy note, the case, work for my phone, the part of the case, the phone in the case, the part, this case, the bottom portion, you, the call on the case, the answer/disconnect on my phone, it, it, amount, it, i, it, the trash]
# 84136   [a good sturdy tpu case with perfect cutouts, no buttons, the product, a tight fit, you, it, awesome case, i, my nexus, the wife, it, she, the case, great deal]
# 189736  [an impressively sturdy case for the samsung galaxy s5 at an incredible price, the case, a great job, the phone, the kickstand, a pleasant and unexpected feature]
# 105362  [a great stylus, i, the rubber tip ones, they, the mesh]
# 21769   [the item, described, i, the protector, it, i, this product, anyone]

# Manually
# 178468  [this case, a month, my resident, this case, a galaxy note 3, the case, my phone, the part of the case, the phone in the case, the part, this case, the bottom portion, you, the call on the case, the answer/disconnect on my phone, it, it, that amount, it, work, i, it, the trash]
# 84136   [a good sturdy tpu case with perfect cutouts, no buttons, the product, a tight fit, you, it, awesome case, i, my nexus, the wife, it, she, the case, great deal]
# 189736  [an impressively sturdy case for the samsung galaxy s5 at an incredible price, the case, a great job, the phone, the kickstand, a pleasant and unexpected feature]
# 105362  [a great stylus, i, the rubber tip ones, they, the mesh]
# 21769   [the item, i, the protector, it, i, this product, anyone]

# @TODO
# Recall + Precision

# Including single noun phrases
noun_phrases = pd.DataFrame.from_dict(Counter(flatten(df['nounPhrases'])), orient='index').\
                reset_index().rename(columns = {'index': 'Noun Phrase', 0: 'Count'}).\
                sort_values(['Count'], ascending = False)
top_20_noun_phrases = noun_phrases.head(20).reset_index().drop(columns = ['index'])
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


noun_phrases_count_map = noun_phrases.set_index('Noun Phrase').to_dict()['Count']

products = df['asin'].value_counts().head(3).index
products_np = df[df['asin'].isin(products)][['asin', 'nounPhrases']].groupby(['asin']).sum().reset_index()
products_np['NPCounts'] = products_np['nounPhrases'].apply(lambda noun_phrases: Counter(noun_phrases).most_common())

# Absolute Count
products_np['abosoluteTop10'] = products_np['NPCounts'].apply(lambda NPCounts: NPCounts[:10])

# Relative counts
products_np['relativeNPFrequency'] = products_np['NPCounts'].apply(lambda NPCounts: [(NPCount[0], NPCount[1]/noun_phrases_count_map[NPCount[0]]) for NPCount in NPCounts])
products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency.sort(key=lambda elem: elem[1], reverse=True))
products_np['relativeTop10'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency[:10])
products_np['relative1'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: [word for word in relativeNPFrequency if word[1] == 1])

product_representative_NPs = products_np[['asin', 'abosoluteTop10', 'relativeTop10', 'relative1']].rename(columns = {'asin': 'productID', 'abosoluteTop10': '10 Representative Noun Phrases (Abosolute)', 'relativeTop10': '10 Representative Noun Phrases (Relative)', 'relative1': 'Representative Noun Phrases (Unique)'})
print(product_representative_NPs)
#     productID  \
# 0  B0042FV2SI
# 1  B005SUHPO6
# 2  B008OHNZI0
#
#                                                                                                                          10 Representative Noun Phrases (Abosolute)  \
# 0  [(it, 1127), (i, 1046), (you, 245), (they, 162), (them, 90), (my phone, 76), (the screen, 73), (me, 53), (this product, 37), (the phone, 35)]
# 1  [(it, 1956), (i, 1884), (you, 448), (this case, 228), (the case, 187), (the phone, 184), (they, 162), (me, 153), (my phone, 148), (he, 128)]
# 2  [(i, 1728), (it, 1319), (you, 464), (they, 251), (me, 126), (the screen, 125), (them, 112), (the screen protector, 73), (the protector, 58), (this product, 56)]
#
#                                                                                                                                                                                                                                                                                         10 Representative Noun Phrases (Relative)
# 0  [(the matte finishing, 1.0), (a deeper scratch on the protector, 1.0), (didnt stay, 1.0), (bubble free surface, 1.0), (the 30th amazing, 1.0), (lights indoors, 1.0), (using screen covers by generic for all long time, 1.0), (that well on my iphone, 1.0), (best investment for any smartphone, 1.0), (sprint 's i4s, 1.0)]
# 1  [(otterbox defender series hybrid case, 1.0), (its quite annoying, 1.0), (charging port keeps, 1.0), (an otterbox for my ipad, 1.0), (the line of the case, 1.0), (love thesei, 1.0), (best case for the iphone, 1.0), (is a fake, 1.0), (this casse, 1.0), (might work for others, 1.0)]
# 2  [(gos:, 1.0), (ta:, 1.0), (the scotch tape method, 1.0), (the high definition, 1.0), (tab number, 1.0), (the tech armor hd clear screen protector, 1.0), (the lint lifter, 1.0), (accomplishment, 1.0), (perfect screen protectors, 1.0), (the home button side, 1.0)]

# Excluding single noun phrases
df['nounPhrases'] = df['nounPhrases'].apply(lambda noun_phrases: [noun_phrase for noun_phrase in noun_phrases if len(noun_phrase.split()) > 1])
noun_phrases = pd.DataFrame.from_dict(Counter(flatten(df['nounPhrases'])), orient='index').\
                reset_index().rename(columns = {'index': 'Noun Phrase', 0: 'Count'}).\
                sort_values(['Count'], ascending = False)
top_20_noun_phrases = noun_phrases.head(20).reset_index().drop(columns = ['index'])
print(top_20_noun_phrases)
#      Noun Phrase  Count
# 0   the phone     38242
# 1   the case      27750
# 2   this case     27186
# 3   my phone      24061
# 4   the price     11420
# 5   this product  10920
# 6   your phone    10012
# 7   the screen    9802
# 8   the battery   8408
# 9   my iphone     8359
# 10  a bit         8038
# 11  the iphone    6440
# 12  this phone    5796
# 13  a lot         5680
# 14  the charger   5346
# 15  the way       5158
# 16  the device    4873
# 17  the time      4468
# 18  the product   4451
# 19  a case        4145

noun_phrases_count_map = noun_phrases.set_index('Noun Phrase').to_dict()['Count']
products = df['asin'].value_counts().head(3).index
products_np = df[df['asin'].isin(products)][['asin', 'nounPhrases']].groupby(['asin']).sum().reset_index()
products_np['NPCounts'] = products_np['nounPhrases'].apply(lambda noun_phrases: Counter(noun_phrases).most_common())

# Absolute Count
products_np['abosoluteTop10'] = products_np['NPCounts'].apply(lambda NPCounts: NPCounts[:10])

# Relative counts
products_np['relativeNPFrequency'] = products_np['NPCounts'].apply(lambda NPCounts: [(NPCount[0], NPCount[1]/noun_phrases_count_map[NPCount[0]]) for NPCount in NPCounts])
products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency.sort(key=lambda elem: elem[1], reverse=True))
products_np['relativeTop10'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: relativeNPFrequency[:10])
products_np['relative1'] = products_np['relativeNPFrequency'].apply(lambda relativeNPFrequency: [word for word in relativeNPFrequency if word[1] == 1])

product_representative_NPs = products_np[['asin', 'abosoluteTop10', 'relativeTop10', 'relative1']].rename(columns = {'asin': 'productID', 'abosoluteTop10': '10 Representative Noun Phrases (Abosolute)', 'relativeTop10': '10 Representative Noun Phrases (Relative)', 'relative1': 'Representative Noun Phrases (Unique)'})
print(product_representative_NPs)
#     productID  \
# 0  B0042FV2SI
# 1  B005SUHPO6
# 2  B008OHNZI0
#
#                                                                                                                                                                       10 Representative Noun Phrases (Abosolute)  \
# 0  [(my phone, 76), (the screen, 73), (this product, 37), (the phone, 35), (the price, 30), (this screen protector, 28), (the screen protector, 24), (my iphone, 24), (the matte finish, 23), (the product, 23)]
# 1  [(this case, 228), (the case, 187), (the phone, 184), (my phone, 148), (your phone, 72), (my iphone, 55), (the iphone, 48), (this product, 47), (the price, 42), (the color, 37)]
# 2  [(the screen, 125), (the screen protector, 73), (the protector, 58), (this product, 56), (tech armor, 53), (the phone, 52), (my iphone, 50), (my phone, 47), (no bubbles, 45), (the price, 44)]
#
#                                                                                                                                                                                                                                                                                                                   10 Representative Noun Phrases (Relative)
# 0  [(the matte finishing, 1.0), (didnt stay, 1.0), (a deeper scratch on the protector, 1.0), (bubble free surface, 1.0), (the 30th amazing, 1.0), (lights indoors, 1.0), (using screen covers by generic for all long time, 1.0), (a clean microfiber cloth/eyeglass cloth, 1.0), (that well on my iphone, 1.0), (best investment for any smartphone, 1.0)]
# 1  [(otterbox defender series hybrid case, 1.0), (its quite annoying, 1.0), (charging port keeps, 1.0), (an otterbox for my ipad, 1.0), (the line of the case, 1.0), (love thesei, 1.0), (best case for the iphone, 1.0), (is a fake, 1.0), (this casse, 1.0), (might work for others, 1.0)]
# 2  [(the scotch tape method, 1.0), (the high definition, 1.0), (tab number, 1.0), (the tech armor hd clear screen protector, 1.0), (the home button side, 1.0), (the lint lifter, 1.0), (perfect screen protectors, 1.0), (hate fingerprints, 1.0), (retinashield screen protector, 1.0), (vs personal phone, 1.0)]






# sentences_df = pd.DataFrame(pd.Series(flatten(df['tokenizedSentences'])), columns = ['sentence']).reset_index().drop(columns = ['index'])
# sentences_df['posTagged'] = sentences_df['sentence'].apply(pos_tag)
# sentences_df['tags'] = sentences_df['posTagged'].apply(lambda posTagged: [tag[1] for tag in posTagged])
# sentences_df.to_csv('data/posTagged.csv', index = False)

# sentences_df = pd.read_csv('data/posTagged.csv')
# sentences_df.head()
# sentences_df['noun_phrases'] = sentences_df['posTagged'].apply(extract_NP)

# sentences_df[sentences_df['tags'].apply(lambda tags: True if "'JJ', 'CC', 'JJ', 'NN'" in tags else False)].head(10)
# sentences_df[sentences_df['tags'].apply(lambda tags: True if "'JJ', 'CC', 'VBG', 'NN'" in tags else False)].head(10)
# sentences_df[sentences_df['tags'].apply(lambda tags: True if "'RBS', 'JJ', 'NN'" in tags else False)].head(10)
# sentences_df[sentences_df['tags'].apply(lambda tags: True if "'RB', 'VBG', 'NN'" in tags else False)].head(10)
# sentences_df[sentences_df['tags'].apply(lambda tags: True if "'NN', 'POS', 'NN'" in tags else False)].head(10)



# class EmojiTokenizer(TreebankWordTokenizer):
#
#     _contractions = MacIntyreContractions()
#     CONTRACTIONS = list(map(re.compile, _contractions.CONTRACTIONS2 + _contractions.CONTRACTIONS3))
#
#     PUNCTUATION = [
#         (re.compile(r'([,])([^\d])'), r' \1 \2'),
#         (re.compile(r'([,])$'), r' \1 '),
#         (re.compile(r'\.\.\.'), r' ... '),
#         (re.compile(r'[;@#$%&]'), r' \g<0> '),
#         # Handles the final period
#         (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
#         (re.compile(r'[?!]'), r' \g<0> '),
#         (re.compile(r"([^'])' "), r"\1 ' "),
#     ]
#
#     emojis = []
#
#     def tokenize(self, text):
#         for regexp, substitution in self.STARTING_QUOTES:
#             text = regexp.sub(substitution, text)
#
#         for regexp, substitution in self.PUNCTUATION:
#             text = regexp.sub(substitution, text)
#
#         text = " " + text + " "
#
#         # split contractions
#         for regexp, substitution in self.ENDING_QUOTES:
#             text = regexp.sub(substitution, text)
#         for regexp in self.CONTRACTIONS:
#             text = regexp.sub(r' \1 \2 ', text)
#
#         # handle emojis
#         for emoticon in list(EMOTICON_RE.finditer(text))[::-1]:
#             self.emojis.append(emoticon.group())
#             pos = emoticon.span()[0]
#             if text[pos - 1] != ' ':
#                 text = text[:pos] + ' ' + text[pos:]
#
#         return text.split()
#
# _replace_html_entities('&#34;')
# tokenize(segment_sent('&#34;100')[0])
#
# tokenizer = EmojiTokenizer()
# for sentence in flatten(df['sentences']):
#     tokenize(sentence, word_tokenizer = tokenizer)
# emojis = tokenizer.emojis
#
# clarence = set(tokenizer.emojis)
#
# zhiwei = {':)': 2520, ':(': 808, ':*': 664, ';)': 427, ':&': 357, ':D': 163, ':[': 152, '=)': 149, ':/': 100, ':>': 57, ':3': 46, ':$': 46, ':P': 36, ':o': 24, '8)': 18, ':-*': 18, ':#': 15, ':o)': 15, ':p': 15, '=]': 14, ':]': 10, ':-))': 9, '=3': 8, ':O': 7, ';D': 7, ':-3': 6, 'D:': 5, ":'(": 5, ':S': 5, '*)': 5, ':-}': 4, ':\\': 4, ':{': 3, ';]': 3, ':c': 3, ':b': 3, ':|': 2, ':^)': 2, '>:(': 2, '3:)': 2, '>:[': 1, ':X': 1, ':L': 1, ':@': 1, ':-0': 1, ':}': 1, '>:)': 1, ":')": 1, '>_>': 1, 'DX': 1}
# zhiwei_set = set(zhiwei.keys())
# clarence = [(':)', 2561), (':(', 667), (':P', 521), ('/8', 512), (':-)', 493), ('):', 482), ('p:', 384), (':D', 345), ('(8', 204), ('8/', 190), (':/', 162), ('=)', 152), ('(:', 143), ('P8', 134), (':-(', 107), ('8)', 79), (':=', 77), ('8:', 77), (':[', 71), (':-D', 57), ('do:', 50), (':-P', 49), ('::', 47), (':p', 42), ('8-p', 31), (']:', 29), ('|8', 27), ('=(', 24), (':-/', 24), ('8p', 24), ('8-P', 19), (':o)', 15), (')8', 14), ('=]', 14), ('D8', 13), (':-p', 13), ('=p', 12), (')=', 11),      ('=P', 11), ('=D', 10), (':]', 9), ('/:', 8), (':d', 7), ('8P', 7), ('=/', 7), ('=-)', 7), ('8D', 7), (':8', 7), ('(=', 6), (':*P', 6), ('=[', 6), (':-d', 6), ('<3', 6), ('D:', 6), (':\\', 5), ('=d', 5), ('8d', 5), ('(-:', 5), (":'(", 5), (':O)', 4), ('>:(', 4), (':-}', 4), (':*)', 3), (':-{', 3), (':*D', 3), ('=o)', 3), (':}', 3), (':-\\', 3), ('=|', 2), (':Op', 2), (':|', 2), ('8-)', 2), ('=}', 2), ('=\\', 2), (':*(', 2), ('8-d', 2), (':-8', 2), ('>:[', 1), (':OP', 1), ('>:)', 1), ('(-8', 1), ('=-[', 1), ('>:|', 1), ('[:', 1), (':*[', 1), (']=', 1), (':{', 1), (':O/', 1), ('[=', 1), (':o/', 1), ('=-D', 1), (":')", 1), (')=>', 1), (':-|', 1), ('/=', 1), ('\\8', 1), (':*d', 1), ('8oP', 1)]
# clarence_set = [pair[0] for pair in clarence]
#
# {emoji: zhiwei[emoji] for emoji in zhiwei_set.difference(clarence_set)}
# len(zhiwei.difference(clarence))
# len(clarence.difference(zhiwei))
# len(clarence.intersection(zhiwei))
# clarence
# zhiwei
# print(Counter(emojis).most_common())
#
# df['sentenceLen'] = df['sentences'].apply(lambda sentences: [len(sen) for sen in sentences])
#
# # set((flatten(sentences_df['tags'])))
# # (Adjective | Noun)* (Noun Preposition)? (Adjective | Noun)* Noun
#
# from nltk.tokenize.treebank import TreebankWordTokenizer
# pos_tag(TreebankWordTokenizer().tokenize("that's"))
#
#
# pos_tag(tokenize(segment_sent('that is')[0]))
# pos_tag(tokenize(segment_sent("that's")[0]))
# pos_tag(tokenize(segment_sent("I'm")[0]))
# pos_tag(tokenize(segment_sent("I am")[0]))
# pos_tag(tokenize(segment_sent("It is")[0]))
# pos_tag(tokenize(segment_sent("It's")[0]))
# pos_tag(tokenize(segment_sent("can't")[0]))
# posTagged = pos_tag(word_tokenize(sent_tokenize('mr. bean')[0]))


# extract_NN(sent)

# from textblob import TextBlob
# df = df[['reviewText']]
# df['blob'] = df['reviewText'].apply(TextBlob)
# df['sentences'] = df['blob'].apply(lambda blob: blob.sentences)
# df['tokens'] = df['sentences'].apply(lambda sentences: [sentence.tokens for sentence in sentences])
# df['tags'] = df['sentences'].apply(lambda sentences: [sentence.tags for sentence in sentences])
# df['noun_phrases'] = df['blob'].apply(lambda blob: blob.noun_phrases)
# df['sentiment'] = df['sentences'].apply(lambda sentences: [sentence.sentiment.polarity for sentence in sentences])
# df.head()
