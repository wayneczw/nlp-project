import json
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nltk import pos_tag


# Load the Dataset
data = []
with open('CellPhoneReview.json') as json_data:
    for l in json_data:
        data.append(json.loads(l))
        
df = pd.DataFrame.from_dict(data)  # load as Dataframe from pandas



# Qn 3.2 part 1 - Popular Products and Frequent Reviewers:
ranked_list_of_reviewers = df['reviewerID'].value_counts()
top10_reviewers = ranked_list_of_reviewers[:10]
print("Top 10 Reviewers:")
print(top10_reviewers)
print()

ranked_list_of_products = df['asin'].value_counts()
top10_products = ranked_list_of_products[:10]
print("Top 10 Products")
print(top10_products)
print()




# Qn 3.2 part 2 - Sentence Segmentation:
nltk.download('punkt')
df_reviewText = df['reviewText']   # Get the 'reviewText' column from the DataFrame
np_reviewText = df_reviewText.values  # get a numpy array with each element being a single review text.

segmented_sentences_for_all_reviews = np.array([sent_tokenize(x) for x in np_reviewText])
sentence_counts_for_all_reviews = np.array([len(x) for x in segmented_sentences_for_all_reviews])
sentence_count_distribution = Counter(sentence_counts_for_all_reviews)
df_graph = pd.DataFrame.from_dict(sentence_count_distribution, orient='index')
df_graph.plot(kind='line')    # Can also try "  kind='bar'  "
plt.show()


#list_review_texts = np_reviewText.tolist() # gives a single string that contains all the review texts.
#combined_string_review_texts = ''.join(list_review_texts)
#tokenized_sentences = sent_tokenize(combined_string_review_texts)

np.random.seed(5)
random_ints = np.random.randint(len(np_reviewText),size=10)
for j in random_ints:
    print("count no. ", j)
    print("Review is: ")
    print(np_reviewText[j])
    print("Tokenized Sentences: ")
    print(segmented_sentences_for_all_reviews[j])




# Qn 3.2 part 3 - Tokenization and Stemming:
tokenized_words_for_all_reviews = np.array([nltk.word_tokenize(x) for x in np_reviewText])
words_counts_for_all_reviews = np.array([len(x) for x in tokenized_words_for_all_reviews])

ps = nltk.stem.PorterStemmer()
def stem_review(review):
    return [ps.stem(x) for x in review]
### Too Inefficient - Limited the tokenized_words_for_all_reviews in the list comprehension to [0:100]
stemmed_words_for_all_reviews = np.array([list(set(stem_review(review))) for review in tokenized_words_for_all_reviews[0:100]])
stemmed_words_counts_for_all_reviews = np.array([len(x) for x in stemmed_words_for_all_reviews])

word_count_distribution = Counter(words_counts_for_all_reviews)
stemmed_word_count_distribution = Counter(stemmed_words_counts_for_all_reviews)
df_graph_no_stem = pd.DataFrame.from_dict(word_count_distribution, orient='index')
df_graph_no_stem.plot(kind='bar', color='DarkBlue')    # Can also try "  kind='line'  "
df_graph_with_stem = pd.DataFrame.from_dict(stemmed_word_count_distribution, orient='index')
df_graph_with_stem.plot(kind='bar', color='DarkOrange')    # Can also try "  kind='line'  "
plt.show()

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
allWordDist = nltk.FreqDist(word.lower() for review in tokenized_words_for_all_reviews for word in review if word not in stopwords)
stemmedWordDist = nltk.FreqDist(word.lower() for review in stemmed_words_for_all_reviews for word in review if word not in stopwords)

all_mostCommon= allWordDist.most_common(20)
stemmed_mostCommon= stemmedWordDist.most_common(20)
print("Before Stemming, Top 20 most common words:")
print(all_mostCommon)
print("After Stemming, Top 20 most common words:")
print(stemmed_mostCommon)



# Qn 3.2 part 4 - POS Tagging:
nltk.download('averaged_perceptron_tagger')
np.random.seed(10)
random_ints = np.random.randint(len(np_reviewText),size=10)
for j in random_ints:
    np.random.seed(8)
    print("count no. ", j)
    print("Chosen Sentence:")
    chosen_review_with_segmented_sentences = segmented_sentences_for_all_reviews[j]
    random_sentence_index = np.random.randint(len(chosen_review_with_segmented_sentences))
    chosen_sentence = chosen_review_with_segmented_sentences[random_sentence_index]
    print(chosen_sentence)
    print("POS-tagged Sentences: ")
    tokenized_sentence = nltk.word_tokenize(chosen_sentence)    
    print(pos_tag(tokenized_sentence))


