import codecs
from turtle import clear
import pandas as pd 
import streamlit as st
import glob
import matplotlib.pyplot as plt
import seaborn as sns 

from unidecode import unidecode
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams

from nltk.stem import SnowballStemmer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis.gensim_models
import streamlit.components.v1 as components

df = pd.read_csv('Data_Transformed\df.csv', low_memory= False)
print(df.head(5))
df['tweet_cleaned'] = df['tweet_cleaned'].apply(lambda x : str(x))

with open('visu_app/lda.html', 'r') as f: 
        html_string = f.read()
with open('visu_app/lda2.html', 'r') as f: 
        html_string2 = f.read()


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
results = st.container()

with header:
    st.title('Covid - 19 et Twitter : Une analyse NLP des tweets français')
    st.text('Dans cette application, nous allons classer les tweets français #Covid de 2021 en utilisant la méthode LDA')

with dataset:
    st.header('Dataset des Tweets 2021')
    st.text('Le scraping a été effectué en utilisant le module snscrape, version dev')
    st.write(df.head(5))

    colonne = st.radio('Choose the column of the Dataframe', ('Tweet', 'Tweet cleané',' Tweet/Tweet cleané'))
    if colonne == 'Tweet':
        st.write(df['tweet'].head(5))
    elif colonne == 'Tweet cleané':
        st.write(df['tweet_cleaned'].head(5))
    else:
        st.write(df[['tweet','tweet_cleaned']].head(5))

    import missingno as msno
    a = msno.matrix(df)
    st.pyplot(a.figure)


with features:
    st.header('Nombre de tweets par mois')
    st.image('visu_app/Répartition par mois.png')
    st.header('Relation entre les mots')
    st.image('visu_app/Bigramme.png')
    


with results:
    st.header('Visualisation du LDA à 3 topics')
    st.components.v1.html(html_string, width = 1300, height = 800)
    st.header('Visualisation du LDA à 20 topics')
    st.components.v1.html(html_string2, width = 1300, height = 800)

    