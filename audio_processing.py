# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

#Importing important libraries
import pandas as pd
import nltk
import boto3

#Reading input files
pics = pd.read_csv('rb_img_name.csv',header=None)
pics_name = pics.iloc[:,0].values

#Opening output files
f = open("rb_pics_output.txt",'a+')

#Function for calling Rekognition
def detect_labels(photo, bucket):

    client=boto3.client('rekognition')

    response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        MaxLabels=15)
    
    for label in response['Labels']:
        print ("Label: " + label['Name'])
        print ("Confidence: " + str(label['Confidence']))
        
        if label['Confidence'] > 80:
            f.write(f" {label['Name']}")
    return len(response['Labels'])

def main():
    for i in range(len(pics)): 
        photo= pics_name[i]
        bucket='robo-input'
        label_count=detect_labels(photo, bucket)      
    f.close()

if __name__ == "__main__":
    main()
    
with open('rb_pics_output.txt','r+') as f:
    records = f.read()
    
unique_word = []
for word in nltk.word_tokenize(records):
    if word not in unique_word:
        unique_word.append(word)
    