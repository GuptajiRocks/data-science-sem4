import pandas as pd
import nltk as nt

d1 = "The Data Mining and Predictive Modelling course"
d2 = "Data Mining course is interesting"

tk = nt.word_tokenize(d1)

print(tk)