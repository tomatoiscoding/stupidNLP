# example connect to mysql

import mysql.connector

cnx = mysql.connector.connect(user = 'root', password = '123456', host = '127.0.0.1', database = 'tw', port = 3306)

cursor = cnx.cursor()

cursor.execute('SET NAMES utf8mb4')
cursor.execute("SET CHARACTER SET utf8mb4")
cursor.execute("SET character_set_connection=utf8mb4")

# read json lines

import json

with open('/Users/bianbeilei/tianchi2017/tweets/exam-twitter.json') as f:
	for line in f:
		tw = json.loads(line)
		cursor.execute("INSERT INTO text (lang, text) VALUES (%s, %s)", (tw['lang'], tw['text']))

cursor.close()

cnx.commit()

cnx.close()

# generate sql query
# filter keywords

for substr in filter[0]:
    sqlString = sqlString+"text LIKE '%"+substr+"%' OR "
    sqlString = sqlString+"text LIKE '%"+substr.upper()+"%' OR "
sqlString=sqlString[:-4]
sqlFilterCommand = "select * from eng where " + sqlString

# load table

cursor.execute("select text from filter")
print(cursor.rowcount)
import codecs
outfile = codecs.open("/Users/bianbeilei/learnNLP/filter.txt", "w", "utf-8")
for row in cursor:
	tmp = ''.join(row)
	tmp = ''.join(tmp.splitlines())
	outfile.write(tmp + "\n")

# clean data

import re

with open('/Users/bianbeilei/stupidNLP/filter.txt') as f:
	content = f.readlines()

myre = [
	r'(?:@[\w_]+)',
	r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)',
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
	r'(?:(?:\d+,?)+(?:\.?\d+)?)',
	r"(?:[a-z][a-z'\-_]+[a-z])",
	r'(?:[\w_]+)',
	r'(?:\S)'
]

tokens_re = re.compile(r'('+'|'.join(myre)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)


# clean review

import pandas as pd

train = pd.read_csv('/Users/bianbeilei/Downloads/rating.csv',header=0)

# train.shape
# train['outputText'][0]

# remove punctuation and numbers

import re

review_letter = re.sub('[^a-zA-Z]', ' ', train['outputText'][0])

# lower case

review_lower = review_letter.lower()
words = review_lower.split()

from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# remove stopwords

words = [w for w in words if not w in stopwords.words("english")]

def clean_review(raw_data):
	review_text = BeautifulSoup(raw_data).get_text()
	review_letters = re.sub("[^a-zA-Z]", " ", review_text)
	words = review_letters.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words))

num_reviews = train["outputText"].size

clean_train_reviews = []

for i in xrange(0, num_reviews):
    clean_train_reviews.append(clean_review(train["outputText"][i]))

# bag of words

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
vocab = vectorizer.get_feature_names()

import numpy as np
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print count, tag

# train word2vec

"""
input is a list of sentences, each sentence is a list of words
First, split a paragraph into sentences.
Second, train word2vec
"""
# raw['summary'][0]
import pandas as pd
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw = pd.read_csv('/Users/bianbeilei/Algeria_summary.csv', header=0)

# raw.shape

def sent_to_word(raw_sentence):
	sent = re.sub("[^a-zA-Z]"," ", raw_sentence)
	words = sent.lower().split()
	return(words)

def sum_to_sent(summary, tokenizer):
	raw_sentences = tokenizer.tokenize(summary.strip())
	sentences = []
	if len(raw_sentences)>0:
		for raw_sentence in raw_sentences:
			single_sentence = sent_to_word(raw_sentence)
			sentences.append(single_sentence)
	return(sentences)

all_sentences = []
for sum in raw["summary"]:
    all_sentences += sum_to_sent(sum, tokenizer)

import logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

model = word2vec.Word2Vec(all_sentences, size=100, min_count = 40,
            window = 10, sample = 0.001)

model.save(model_name)
