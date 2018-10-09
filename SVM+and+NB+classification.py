
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer

import csv

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_validate

from sklearn import svm

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import classification_report


# In[2]:


#TF-IDF Matrix
with open('C:/Users/rajpu/Desktop/Web analytics Final/final_data_News_Stock_quotes.csv', 'r') as f:
    data = [tuple(line) for line in csv.reader(f)]
    
Lable,Date,News,Company,Sentiment,Open,High,Low,Close,Volume=zip(*data)

date=list(Date)
news=list(News)
sent=list(Sentiment)
lab=list(Lable)

tfidf_vect = TfidfVectorizer() 
tfidf_vect = TfidfVectorizer(stop_words="english") 

dtm= tfidf_vect.fit_transform(news)

print("type of dtm:", type(dtm))
print("size of tfidf matrix:", dtm.shape)


# In[3]:


print("type of vocabulary:", type(tfidf_vect.vocabulary_))
print("index of word 'city' in vocabulary:",       tfidf_vect.vocabulary_['city'])
voc_lookup={tfidf_vect.vocabulary_[word]:word             for word in tfidf_vect.vocabulary_}
print("total number of words:", len(voc_lookup ))

print("\nOriginal text: \n"+news[1])

print("\ntfidf weights: \n")
doc0=dtm[0].toarray()[0]
print(doc0.shape)


top_words=(doc0.argsort())[::-1][0:20]
print([(voc_lookup[i], doc0[i]) for i in top_words])


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(                dtm, sent, test_size=0.3, random_state=0)
# train a multinomial naive Bayes model using the testing data
clf = MultinomialNB().fit(X_train, y_train)

predicted=clf.predict(X_test)

labels=sorted(list(set(sent)))

precision, recall, fscore, support=     precision_recall_fscore_support(     y_test, predicted, labels=labels)
    
#print(precision)
#print(recall)
#print(fscore)
#print(support)

print(classification_report(y_test, predicted, target_names=labels))





# In[5]:


metrics = ['precision_macro', 'recall_macro', "f1_macro"]

clf = MultinomialNB()
#clf = MultinomialNB(alpha=0.8)

cv = cross_validate(clf, dtm, sent, scoring=metrics, cv=5)
print("Test data set average precision:")
print(cv['test_precision_macro'])
print("\nTest data set average recall:")
print(cv['test_recall_macro'])
print("\nTest data set average fscore:")
print(cv['test_f1_macro'])

# To see the performance of training data set use 
#cv['train_xx_macro']
print("\n Train data set average fscore:")
print(cv['train_f1_macro'])


# In[6]:


#SVM model

metrics = ['precision_macro', 'recall_macro', "f1_macro"]

# initiate an linear SVM model
clf = svm.LinearSVC()

cv = cross_validate(clf, dtm, sent, scoring=metrics, cv=5)
print("Test data set average precision:")
print(cv['test_precision_macro'])
print("\nTest data set average recall:")
print(cv['test_recall_macro'])
print("\nTest data set average fscore:")
print(cv['test_f1_macro'])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(                dtm, sent, test_size=0.3, random_state=0)
# train a multinomial naive Bayes model using the testing data
#clf = MultinomialNB().fit(X_train, y_train)
clf = svm.LinearSVC().fit(X_train, y_train)
predicted=clf.predict(X_test)

labels=sorted(list(set(sent)))

precision, recall, fscore, support=     precision_recall_fscore_support(     y_test, predicted, labels=labels)
    
#print(precision)
#print(recall)
#print(fscore)
#print(support)

print(classification_report(y_test, predicted, target_names=labels))




