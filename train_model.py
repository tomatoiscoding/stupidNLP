import pandas as pd
import logging
from gensim.models import word2vec
import nltk.data
import numpy as np
import re

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw = pd.read_csv('/Users/bianbeilei/stupidNLP/data/review1.csv', header = 0, encoding = 'ISO-8859-1')

all_sentences = []
for sum in raw["review"]:
    all_sentences += sum_to_sent(sum, tokenizer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec(all_sentences, size=300, min_count = 1, window = 10, sample = 0.001)

model.save(model_name)

# num_features = 100

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter = counter + 1
    return reviewFeatureVecs

# remove stop words
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(clean_review(review, remove_stopwords = True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)# trainDataVecs.shape

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(clean_review(review, remove_stopwords = True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
# random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(trainDataVecs, train["star"])
result = forest.predict(testDataVecs)

from sklearn import metrics

print(metrics.classification_report(test['star'],result))

# bag of words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis = 0)
forest = forest.fit(train_data_features, train["star"])
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)


