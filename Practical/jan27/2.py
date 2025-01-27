import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import math
import time
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)

text = """Energy-efficient building designs and sustainable materials are crucial for reducing carbon emissions. 
Passive solar designs and advanced insulation systems help minimize energy demands. 
Retrofits like LED lighting and smart HVAC systems are equally impactful."""

sentences = sent_tokenize(text)

stoploss = set(stopwords.words("english"))

f_matrix = {}
for sentence in sentences:
    ws = word_tokenize(sentence.lower())
    filtered_words = [word for word in ws if word.isalnum() and word not in stoploss]
    ftable = defaultdict(int)
    for word in filtered_words:
        ftable[word] += 1
    f_matrix[sentence] = ftable


tf_matrix = {}
for sentence, freq_table in f_matrix.items():
    tf_table = {}
    total_words = sum(freq_table.values())
    for word, freq in freq_table.items():
        tf_table[word] = round((freq / total_words), 3)
    tf_matrix[sentence] = tf_table

word_document_freq = defaultdict(int)
for sentence, freq_table in f_matrix.items():
    for word in freq_table:
        word_document_freq[word] += 1

idf_matrix = {}
total_documents = len(sentences)
for word, count in word_document_freq.items():
    idf_matrix[word] = round((math.log10(total_documents / float(count))), 2)

tf_idf_matrix = {}
for sentence, tf_table in tf_matrix.items():
    tf_idf_table = {}
    for word, tf_value in tf_table.items():
        tf_idf_table[word] = round((tf_value * idf_matrix[word]), 3)
    tf_idf_matrix[sentence] = tf_idf_table

for i in range(len(sentences)):
    print(pd.Series(tf_matrix).iloc[i])

for i in range(len(sentences)):
    print(pd.Series(idf_matrix).iloc[i])

for i in range(len(sentences)):
    print(pd.Series(tf_idf_matrix).iloc[i])

