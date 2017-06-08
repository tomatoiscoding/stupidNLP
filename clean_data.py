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

