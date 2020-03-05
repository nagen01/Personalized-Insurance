# -*- coding: utf-8 -*-
"""
Spyder Editor
This program fatch input from image/text and predict which insurance to be suggested
"""

#Importing important libraries
import pandas as pd
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
import boto3
#import json
import pickle

#Reading input files
pics = pd.read_csv('rb_image_name.csv',header=None)
pics_name = pics.iloc[:,0].values

#Opening output files
f = open("rb_image_output.txt",'a+')

#Function for calling Rekognition: impage processing and getting label
def detect_labels(photo, bucket):
    
    client=boto3.client('rekognition')
    response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        MaxLabels=15)
    
    for label in response['Labels']:
        print ("Label: " + label['Name'])
        print ("Confidence: " + str(label['Confidence']))
        
        if label['Confidence'] > 80:
            f.write(f" {label['Name'].lower()}")
    return len(response['Labels'])

#function for Text processing:
with open("input_text_file.txt","r+") as k:
    sentences = k.read()
def text_to_word(text):
    text = re.sub(r"\[[0-9]*\]",' ',text)
    text = re.sub(r"\s+",' ',text)
    
    clean_text = re.sub(r"\W",' ',text)
    clean_text = re.sub(r"\d",' ',clean_text)
    clean_text = clean_text.lower()
    clean_text = re.sub(r"\s+",' ',clean_text)  
    
    sentences_list = nltk.sent_tokenize(clean_text)
    
    return sentences_list
#Main function and calling
def main():
    for i in range(len(pics)):
        print(pics_name[i])
        photo= pics_name[i]
        bucket='robo-input'
        detect_labels(photo, bucket)    
    sentences_list = text_to_word(sentences)
    for sentence in sentences_list:
        words = nltk.word_tokenize(sentence)
        for word in words:
            f.write(f" {word}")
    f.close()

if __name__ == "__main__":
    main()

#Openning the word collection file for further processing
with open('rb_image_output.txt','r+') as f:
    records = f.read()
    records = records.lower()

#Removing stopword and finding unique words for further processing  
stop_words = nltk.corpus.stopwords.words('english')
unique_word = []
u_len = 0
for word in nltk.word_tokenize(records):
    if word not in stop_words:
        if word not in unique_word:
            if len(word) >= 3:
                u_len += len(word)
                unique_word.append(word)

#function Calling comprehend for sentiment evaluation and risk calculation: 
def sentiment_evaluation(text):

    comprehend = boto3.client(service_name='comprehend', region_name='eu-west-1')
    
    print('Calling DetectSentiment')
    sentiment_dump = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return sentiment_dump
"""    
    positive_score = sentiment_dump['SentimentScore']['Positive']
    negative_score = sentiment_dump['SentimentScore']['Negative']
    neutral_score = sentiment_dump['SentimentScore']['Neutral']
    mixed_score = sentiment_dump['SentimentScore']['Mixed']
    
    print(f"positive_sentiment_score : {positive_score}")
    print(f"negative_sentiment_score : {negative_score}")
    print(f"neutral_sentiment_score : {neutral_score}")
    print(f"mixed_sentiment_score : {mixed_score}")
    
    print('End of DetectSentiment\n')
   
    return sentiment_dump
"""
full_len = 0
no_comprehend_call = 0
positive_sentiment_score = 0
negative_sentiment_score = 0
neutral_sentiment_score = 0
mixed_sentiment_score = 0
for word in unique_word:
    full_len += len(word)
    text_len = 0
    text_len += len(word)

    if text_len >= 4500 or full_len == u_len:        
        sentiment_dump = sentiment_evaluation(text)
        no_comprehend_call += 1
        positive_sentiment_score += sentiment_dump['SentimentScore']['Positive']
        negative_sentiment_score += sentiment_dump['SentimentScore']['Negative']
        neutral_sentiment_score += sentiment_dump['SentimentScore']['Neutral']
        mixed_sentiment_score += sentiment_dump['SentimentScore']['Mixed'] 
        
        print(f"{no_comprehend_call}th comprehend call sentiment score:")
        print(f"positive_sentiment_score : {sentiment_dump['SentimentScore']['Positive']}")
        print(f"negative_sentiment_score : {sentiment_dump['SentimentScore']['Negative']}")
        print(f"neutral_sentiment_score : {sentiment_dump['SentimentScore']['Neutral']}")
        print(f"mixed_sentiment_score : {sentiment_dump['SentimentScore']['Mixed']}")
        print(f"End of {no_comprehend_call}th comprehend call sentiment score:")
    else:
        text = ' '.join(word for word in unique_word)

positive_sentiment_score = positive_sentiment_score/no_comprehend_call
negative_sentiment_score = negative_sentiment_score/no_comprehend_call
neutral_sentiment_score = neutral_sentiment_score/no_comprehend_call
mixed_sentiment_score = mixed_sentiment_score/no_comprehend_call

print(f"no_comprehend_call : {no_comprehend_call}")
print('Final sentiment scores')
print(f"positive_sentiment_score : {positive_sentiment_score}")
print(f"negative_sentiment_score : {negative_sentiment_score}")
print(f"neutral_sentiment_score : {neutral_sentiment_score}")
print(f"mixed_sentiment_score : {mixed_sentiment_score}")
print('End of final sentiment scores')
#Checking for -ve sentiment score and deciding based on the result for further processing:

if negative_sentiment_score > 0.5:
    print('Person is risky, stopping processing')
else:
    #Loading model for associated words:               
    with open('association_model.pickle','rb') as f:
        fsw = pickle.load(f)
        
    #Finding final word list alog with associated words
    final_wl = []  
    for w in unique_word:
        try:
            msw = fsw.wv.most_similar(w)
            if w not in final_wl:
                final_wl.append(w)            
            for i in range(10):
                if msw[i][0] not in final_wl:
                    final_wl.append(msw[i][0])
        except KeyError:
            print(f"Oops! {w} is not found in the list")        
    #
    print("Words collected from image and text:")
    print(records)
    print("\n")
    print("Unique_words:")
    print(unique_word)
    print("\n")
    print("Final Associated words:")
    print(final_wl)
    print("\n")
    #Getting Health Insurence related words:
    health_words = pd.read_csv('health_insurence_words_list.csv', header = None)
    health_words = health_words.values.tolist()
    #print(type(health_words))
    print("Health Words:")
    print(health_words[0])
    health_match_word = [word for word in final_wl if word in health_words[0]]
    print("\n")
    print("Health_match_words:")
    print(health_match_word)
    #print(type(health_match_word))
    print(f"health_words: {len(health_words[0])}")
    print(f"final user input to Health model: {len(final_wl)}")
    print(f"health_match_word: {len(health_match_word)}")    
    #print(f"Health Insurance % wrt Health Dictionary: {len(health_match_word)/len(health_words[0])}")
    print(f"Health Insurance % wrt customer input words: {len(health_match_word)/len(final_wl)}")
    print("\n")
    
    car_words = pd.read_csv('car_insurence_words_list.csv', header = None)
    car_words = car_words.values.tolist()
    #print(type(car_words))
    print("Car Words:")
    print(car_words[0])
    car_match_word = [word for word in final_wl if word in car_words[0]]
    print("\n")
    print("car_match_words:")
    print(car_match_word)
    #print(type(car_match_word))
    print(f"car_words: {len(car_words[0])}")
    print(f"final user input to Car model: {len(final_wl)}")
    print(f"car_match_word: {len(car_match_word)}")
    #print(f"Car Insurance % wrt Car Dictionary: {len(car_match_word)/len(car_words[0])}")
    print(f"Car Insurance % wrt customer input words: {len(car_match_word)/len(final_wl)}")
    
