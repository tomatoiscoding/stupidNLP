import pandas as pd
import logging
from gensim.models import word2vec
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw = pd.read_csv('/Users/bianbeilei/stupidNLP/data/review.csv', header = 0, encoding = 'utf-8')

all_sentences = []
for sum in raw["review"]:
    all_sentences += sum_to_sent(sum, tokenizer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec(all_sentences, size=100, min_count = 40, window = 10, sample = 0.001)

model.save(model_name)