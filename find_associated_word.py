# -*- coding: utf-8 -*-
"""
Word2Vec model for association and collection of words from the links for specific insurance 
Created on Sat Aug 24 16:50:36 2019

@author: Nagendra
"""
#Importing important libraries
import pandas as pd
import nltk
nltk.download('stopwords')
import re
from gensim.models import Word2Vec
import bs4 as bs
import urllib.request
import pickle

#Finding words from multiple url links
def urls_words(url_links):
    text = ""
    for i in range(len(url_links)): 
        try:
            source = urllib.request.urlopen(url_links[i,0]).read()
            soup = bs.BeautifulSoup(source,'lxml')
            for paragraph in soup.find_all('p'):
                text += paragraph.text
            print(url_links[i,0])
        except urllib.error.URLError:
            print('/n')
            print(f"Ops! {url_links[i,0]} not opened")
            print('/n')
        
    text = re.sub(r"[\[0-9]*\]"," ",text)
    text = re.sub(r"\s+"," ",text)
    
    text = text.lower()
    text = re.sub(r"[@#$%^&\*\.\-\?\<\>\:\;\'\"\/\(\)\!\~\,]"," ",text)
    text = re.sub(r"\W"," ",text)
    text = re.sub(r"\d"," ",text)
    text = re.sub(r"\s+"," ",text)
    
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences     

stop_words = nltk.corpus.stopwords.words('english')
#Collecting all type of insurance related sentences/words and making Word2Vec model for finding associated words
All_insurance_url = pd.read_csv('All_insurance_links.csv',header = None)
all_url = All_insurance_url.values
all_words = urls_words(all_url)
for i in range(len(all_words)):
    all_words[i] = [word for word in all_words[i] if word not in stop_words]

#Model preparation
model = Word2Vec(all_words,min_count=1)

#Saving model for future use
association_model = model
with open('association_model.pickle','wb') as f:
    pickle.dump(association_model,f)

#Function for specific insurance related word extraction from links and storing into the specific file
def specific_insurance_words(spec_insur_words, x):
    unique_word = []
    u_word_count = 0
    for i in range(len(spec_insur_words)):
        for word in spec_insur_words[i]:
            print(word)
            if word not in stop_words:
                if word not in unique_word:
                    if len(word) >= 3:
                        u_word_count += 1
                        unique_word.append(word)
                        x.write(f"{word},")
    #print(unique_word)
    print(unique_word)
                   
#Collection of health insurance related words from links    
health_insurance_url = pd.read_csv('health_insurance_links.csv',header = None)
health_url = health_insurance_url.values
health_words = urls_words(health_url)
h = open('health_insurence_words_list.csv','w+')
specific_insurance_words(health_words, h)
h.close

#Collection of Car insurance related words from links
car_insurance_url = pd.read_csv('car_insurance_links.csv',header = None)
car_url = car_insurance_url.values
print(car_url)
car_words = urls_words(car_url)
print(car_words)
c = open('car_insurence_words_list.csv','w+')
specific_insurance_words(car_words, c)
c.close