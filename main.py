import os
import re
import csv
import sys
import json
import argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import keras
from keras.utils import np_utils
from keras.models import load_model

import model


word_mapping={}
def parse_data(X_data):
    global word_mapping
    # Zero Padding to max_len
    max_len = 30
    # Transform sentences into sequence of index
    
    for index in range(len(X_data)):
        tmp=[]
        X_data[index]=X_data[index].split(" ")
        for word in X_data[index]:
            if word in word_mapping:
                #tmp.append(np.add(word_mapping[word][:100],word_mapping[word][100:]))
                tmp.append(np.float32(word_mapping[word]))
            else:
                tmp.append(np.zeros(200,dtype=np.float32))
        if len(tmp)<max_len:
            for ext in range(max_len-len(tmp)):
                tmp.append(np.zeros(200,dtype=np.float32))
        elif len(tmp)>max_len:
            tmp=tmp[:max_len]
        tmp=np.array(tmp,dtype=np.float32)
        X_data[index]=tmp
    print('finished parsing!')
    return X_data
    






def load_data():
    X_train, Y_train=[],[]
    with open('data/train.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        index=0
        for row in reader:
            if index==0:
                index+=1
                continue
            X_train.append(row)
            index+=1
        X_train=np.array(X_train)
        Y_train=X_train[:,3]
        X_train=X_train[:,2]
    X_test=[]
    with open('data/test.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        index=0
        for row in reader:
            if index==0:
                index+=1
                continue
            X_test.append(row)
            index+=1
        X_test=np.array(X_test)
        X_test_ID=X_test[:,0]
        X_test=X_test[:,2]
    
    
    
    return X_train.tolist(),Y_train.tolist(),X_test.tolist(),X_test_ID.tolist()
    
def infer(X_test,X_test_ID):
    model=load_model("LSTMv9.model")
    result=model.predict(X_test)
    result=np.argmax(result,axis=1)
    with open("105502040v9.csv",'w') as res:
        writer=csv.writer(res)
        writer.writerow(["PhraseId","Sentiment"])
        for index in range(result.shape[0]):
            writer.writerow([X_test_ID[index],result[index]])
            
    return
    


def main(opt):
    
    #load data
    X_train,Y_train,X_test,X_test_ID=load_data()
    #word_mapping={}
    word_indexing=0
   
    w2vmodel = Word2Vec.load(
'w2v/word2vec_model/wiki.en.text.model')
    print("read model successful")
    for k, v in w2vmodel.wv.vocab.items():
        word_mapping[k]= w2vmodel.wv[k]
    del w2vmodel
    
    
    if opt.train:
        X_train=parse_data(X_train)
        X_train = np.array(X_train,dtype=np.float32)
        Y_train = np.array(Y_train,dtype=np.int8)
        Y_train = np_utils.to_categorical(Y_train, num_classes=5)
        model.train(X_train,Y_train)
    elif opt.infer:
        X_test = parse_data(X_test)
        X_test = np.array(X_test,dtype=np.float32)
        infer(X_test,X_test_ID)
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tech Platform Deep Learning Version 2')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    opts = parser.parse_args()
    
    
    main(opts)
