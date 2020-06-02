# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:44:34 2020

@author: roy
"""

#word embedding with krish naik
# from tensorflow>2.0 keras is integrated within
from tensorflow import one_hot
from keras.preprocessing.text import one_hot
sent=['the glass of milk',
      'the glass of juice',
      'the cup of tea',
      'i am a good boy',
      'i am a good developer',
      'understand the meaning of words',
      'your videos are good'
      
      ]
      
voc_size=int(10e3)

#getting the index out of 10e3 from the dictionary voc_size
onehot_repr=[one_hot(words,voc_size) for words in sent]
print(onehot_repr)

#word embedding representation
from tensorflow.python.keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences #making sure the sentences are of equal size

from tensorflow.python.keras import Sequential #needed for the embedding
import numpy as np

sent_length=8 #set the max sent length
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

dim=10# how many features

#adding embedding layer to the sequential model
model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile()
model.summary()

#see how the words got converted
model.predict(embedded_docs).shape
embedded_docs[10]

model.predict(embedded_docs)[0] #the 8 words; for each word, a vector of 10 floats

