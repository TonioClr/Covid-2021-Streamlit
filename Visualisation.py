
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
df['tweet_cleaned'] = df['tweet_cleaned'].apply(lambda x : str(x))

sns.countplot(data = df, x = 'months', palette = 'Set1')
plt.savefig('Répartition par mois.png')

list_tweet = df['tweet_cleaned'].tolist()
list_token = ' '.join(list_tweet).split()
list_bigrams = list(ngrams(list_token, 2))
counter = Counter(list_bigrams)
df_plot = pd.DataFrame(counter.most_common(20))
fig, ax = plt.subplots()
ax = sns.barplot(x=df_plot[1], y=df_plot[0], palette = 'Set1').set_title('Bigramme - TOP 20, Covid exclu')
plt.savefig('Bigramme.png')