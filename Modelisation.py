
from turtle import clear
import pandas as pd 
import streamlit as st
import glob
import matplotlib.pyplot as plt
import seaborn as sns 


import re
import nltk
from collections import Counter
from nltk.util import ngrams

from nltk.stem import SnowballStemmer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis.gensim_models


df = pd.read_csv('Data_Transformed\df.csv', low_memory= False)
print(df.head(5))
df['tweet_cleaned'] = df['tweet_cleaned'].apply(lambda x : str(x))

list_tweet = df['tweet_cleaned'].tolist()
list_token = ' '.join(list_tweet).split()
list_bigrams = list(ngrams(list_token, 2))
counter = Counter(list_bigrams)
df_plot = pd.DataFrame(counter.most_common(20))
fig, ax = plt.subplots()
ax = sns.barplot(x=df_plot[1], y=df_plot[0], palette = 'Set1').set_title('Bigramme - TOP 20, Covid exclu')
plt.savefig('Bigramme.png')

##Entrainement du modèle
list_tweet = df['tweet_cleaned'].tolist()
mots = [d.split() for d in list_tweet]
id2word = corpora.Dictionary(mots)
corpus = []
for text in mots:
    new = id2word.doc2bow(text)
    corpus.append(new)

lda_model_gen = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics = 3)
vis = pyLDAvis.gensim_models.prepare(lda_model_gen, corpus, id2word)
lda_model_gen_2 = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics = 20)
vis2 = pyLDAvis.gensim_models.prepare(lda_model_gen_2, corpus, id2word)


html_string = pyLDAvis.prepared_data_to_html(vis)
pyLDAvis.save_html(vis, 'lda.html')
html_string_2 = pyLDAvis.prepared_data_to_html(vis2)
pyLDAvis.save_html(vis2, 'lda2.html')



 

