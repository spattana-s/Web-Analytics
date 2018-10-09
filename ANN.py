
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


# Exercise 5.1: Load data

import pandas as pd
import nltk,string
from gensim import corpora

data=pd.read_csv("C:/Users/rajpu/Desktop/Web analytics Final/final_data2.csv", header=0,encoding='Latin1', delimiter=",")
data.head()
len(data)

# if your computer does not have enough resource
# reduce the dataset
data=data.loc[0:237]


# In[5]:


# Exercise 5.2 Prepocessing data: Tokenize, pad sentences

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
 
import numpy as np

# set the maximum number of words to be used
MAX_NB_WORDS=10000

# set sentence/document length
MAX_DOC_LEN=500

# get a Keras tokenizer
# https://keras.io/preprocessing/text/
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data["News"])

# convert each document to a list of word index as a sequence
sequences = tokenizer.texts_to_sequences(data["News"])

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_sequences = pad_sequences(sequences,                                  maxlen=MAX_DOC_LEN,                                  padding='post',                                  truncating='post')

print(padded_sequences[0])


# In[6]:


# get the mapping between word and its index
tokenizer.word_index['trump']

# get the count of each word
tokenizer.word_counts['trump']


# In[7]:


# Split data for training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(                        padded_sequences, data['Sentiment'],                        test_size=0.3, random_state=1)


# In[8]:


# Exercise 5.3: Create CNN model

from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model

# The dimension for embedding
EMBEDDING_DIM=100

# define input layer, where a sentence represented as
# 1 dimension array with integers
main_input = Input(shape=(MAX_DOC_LEN,),                    dtype='int32', name='main_input')

# define the embedding layer
# input_dim is the size of all words +1
# where 1 is for the padding symbol
# output_dim is the word vector dimension
# input_length is the max. length of a document
# input to embedding layer is the "main_input" layer
embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                     output_dim=EMBEDDING_DIM,                     input_length=MAX_DOC_LEN,                    name='embedding')(main_input)


# define 1D convolution layer
# 64 filters are used
# a filter slides through each word (kernel_size=1)
# input to this layer is the embedding layer
conv1d_1= Conv1D(filters=64, kernel_size=1,                  name='conv_unigram',                 activation='relu')(embed_1)

# define a 1-dimension MaxPooling 
# to take the output of the previous convolution layer
# the convolution layer produce 
# MAX_DOC_LEN-1+1 values as ouput (???)
pool_1 = MaxPooling1D(MAX_DOC_LEN-1+1,                       name='pool_unigram')(conv1d_1)

# The pooling layer creates output 
# in the size of (# of sample, 1, 64)  
# remove one dimension since the size is 1
flat_1 = Flatten(name='flat_unigram')(pool_1)

# following the same logic to define 
# filters for bigram
conv1d_2= Conv1D(filters=64, kernel_size=2,                  name='conv_bigram',activation='relu')(embed_1)
pool_2 = MaxPooling1D(MAX_DOC_LEN-2+1, name='pool_bigram')(conv1d_2)
flat_2 = Flatten(name='flat_bigram')(pool_2)

# filters for trigram
conv1d_3= Conv1D(filters=64, kernel_size=3,                  name='conv_trigram',activation='relu')(embed_1)
pool_3 = MaxPooling1D(MAX_DOC_LEN-3+1, name='pool_trigram')(conv1d_3)
flat_3 = Flatten(name='flat_trigram')(pool_3)

# Concatenate flattened output
z=Concatenate(name='concate')([flat_1, flat_2, flat_3])

# Create a dropout layer
# In each iteration only 50% units are turned on
drop_1=Dropout(rate=0.5, name='dropout')(z)

# Create a dense layer
dense_1 = Dense(192, activation='relu', name='dense')(drop_1)
# Create the output layer
preds = Dense(1, activation='sigmoid', name='output')(dense_1)

# create the model with input layer
# and the output layer
model = Model(inputs=main_input, outputs=preds)


# In[9]:


# Exercise 5.4: Show model configuration

model.summary()
#model.get_config()
#model.get_weights()
#from keras.utils import plot_model
#plot_model(model, to_file='cnn_model.png')


# In[10]:


# Exercise 5.4: Compile the model

model.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])


# In[11]:


# Exercise 5.5: Fit the model

BATCH_SIZE = 64
NUM_EPOCHES = 10

# fit the model and save fitting history to "training"
training=model.fit(X_train, y_train,                    batch_size=BATCH_SIZE,                    epochs=NUM_EPOCHES,                   validation_data=[X_test, y_test],                    verbose=2)


# In[12]:


# Exercise 5.6. Investigate the training process

import matplotlib.pyplot as plt
import pandas as pd
# plot a figure with size 20x8

# the fitting history is saved as dictionary
# covert the dictionary to dataframe
df=pd.DataFrame.from_dict(training.history)
df.columns=["train_acc", "train_loss",             "val_acc", "val_loss"]
df.index.name='epoch'
print(df)

# plot training history
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3));

df[["train_acc", "val_acc"]].plot(ax=axes[0]);
df[["train_loss", "val_loss"]].plot(ax=axes[1]);
plt.show();


# In[13]:


# Exercise 5.6: Use early stopping to find the best model

from keras.callbacks import EarlyStopping, ModelCheckpoint

# the file path to save best model
BEST_MODEL_FILEPATH="best_model"

# define early stopping based on validation loss
# if validation loss is not improved in 
# an iteration compared with the previous one, 
# stop training (i.e. patience=0). 
# mode='min' indicate the loss needs to decrease 
earlyStopping=EarlyStopping(monitor='val_loss',                             patience=0, verbose=2,                             mode='min')

# define checkpoint to save best model
# which has max. validation acc
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH,                              monitor='val_acc',                              verbose=2,                              save_best_only=True,                              mode='max')

# compile model
model.compile(loss="binary_crossentropy",               optimizer="adam", metrics=["accuracy"])

# fit the model with earlystopping and checkpoint
# as callbacks (functions that are executed as soon as 
# an asynchronous thread is completed)
model.fit(X_train, y_train,           batch_size=BATCH_SIZE, epochs=NUM_EPOCHES,           callbacks=[earlyStopping, checkpoint],
          validation_data=[X_test, y_test],\
          verbose=2)


# In[14]:


# Exercise 5.7: Load the best model

# load the model using the save file
model.load_weights("best_model")

# predict
pred=model.predict(X_test)
print(pred[0:5])
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

