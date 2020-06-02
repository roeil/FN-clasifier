# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:03:58 2020

@author: roy
"""

#detect fake news; KAGGLE
import pandas as pd
df=pd.read_csv('fake-news/train.csv')
df.head()

x=df.drop('label',axis=1) #pandas function; axis=1 means column
y=df['label']

df.shape

from sklearn.feature_exrtaction.text import CountVectorizer, TfidVectorizer, HashingVectorizer

df=df.dropna()

meassages=df.copy()

messages.reset_index(inplace=True) #after dropping some of the indices will get dropped; indices will lose order. this restores the order


#emove stop words and special characters, and stemming

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus=[]
for i in range(0, len(messages)):
    review = re.sub('[^az-A-Z]','',messages['title'][i]) #remove special characters
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=''.join(review)
    corpus.append(review)
    
    
#apply Bag of Words
#applying CountVectorizer
# Creating the BOW model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5e3,ngram_range=(1,3)) #ngram means taking one, two, and 3 words...
x = cv.fit_transform(corpus).toarray()


#output feature
y=messages['label']

#divide DS into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=0)
cv.get_feature_names()[:20]

cv.get_params()

count_df=pd.DataFrame(X_train,columns=cv.get_feature_names())
count_df.head()


def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arrange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
        
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
        
        
#multinomial naive bayes algorithm
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools


classifier.fit(X_tain, y_train)
pred=classifier.predict(X_test)
score = metric.accuracy_score(y_test,pred)
print("accuracy: %0.3f",% score))
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm, classes=['fake','real'])

pred=classifier.predict(x_test)
score=metrics.accuracy_score(y_test,pred)
y_train.shape

#passive aggressive classifier algorithm - works well with text data
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)


linear_clf.fit(x_train,y_train)
pred=linear_clf.predict(X_test)

score=metrics.accuracy_score(y_test,pred)

print("accuracy: %0.3f" %score)

cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cn,classes=['fake data','real data'])



#multinomial classifier with hyperparameter

classifier=MultinomialNB(alpha=0.1) #initialize

previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(aplha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score=metrics.accuracy_score(y_test,y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("alpha: {},score:{}".format(alpha,score))

#get feature names
feature_names=cv.get_feature_names()
classifier.coef_[0]



## most real
sorted(zip(classifier.coef_[0],feature_names), reverse=True)[:20]



#second part
#same as above but for: messages['title']           
#apply tf/idf
#from line 42, instead of BOW, tf/idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5e3,ngram_range=(1,3))# we can take (1,2) this is better
X=tfidf_v.fit_transform(corpus).toarray()

X.shape
#then from line 52
# line 59
tfidf_v.get_feature_names()[:20]
tfidf_v.get_params()
count_df=pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())
count_df.head()# these are all the X independent features
#then iline 93 multinmomial NB algorithm

#then
#hashing vectorizer
hs_vec=HashingVectorizer(n_faetures=5e3,non_negative=True)
X=hs_vectorizer.fit_transform(corpus).toarray()
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=0)
