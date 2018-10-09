
# coding: utf-8

# In[1]:


import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime

consumer_key = 'Nt9OPpV3YTsRkFlUdzRclki6Y'
consumer_secret = 'qonyyVaw6ISlHa1cHrQoiug9iBEJrjqQfv2J6tveaPSyycapxl'
access_token = '995853085-85KRxkGxhdBPkF9cTs4oEJS6uWhxMj4ZZTndv9M0'
access_secret = '41R90OY0tHa7isHU9ozdLSS7OXNjojl3x28DpCr5GC1Kv'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
class MyListener(StreamListener):
    
    # constructor
    def __init__(self, output_file, time_limit):
        
            # attribute to get listener start time
            self.start_time=datetime.datetime.now()
            
            # attribute to set time limit for listening
            self.time_limit=time_limit
            
            # attribute to set the output file
            self.output_file=output_file
            
            # initiate superclass's constructor
            StreamListener.__init__(self)
    
    # on_data is invoked when a tweet comes in
    # overwrite this method inheritted from superclass
    # when a tweet comes in, the tweet is passed as "data"
    def on_data(self, data):
        
        # get running time
        running_time=datetime.datetime.now()-self.start_time
        print(running_time)
        
        # check if running time is over time_limit
        if running_time.seconds/60.0<self.time_limit:
            
            # ***Exception handling*** 
            # If an error is encountered, 
            # a try block code execution is stopped and transferred
            # down to the except block. 
            # If there is no error, "except" block is ignored
            try:
                # open file in "append" mode
                with open(self.output_file, 'a') as f:
                    # Write tweet string (in JSON format) into a file
                    f.write(data)
                    
                    # continue listening
                    return True
                
            # if an error is encountered
            # print out the error message and continue listening
            
            except BaseException as e:
                print("Error on_data:" , str(e))
                
                # if return "True", the listener continues
                return True
            
        else:  # timeout, return False to stop the listener
            print("time out")
            return False
 
    # on_error is invoked if there is anything wrong with the listener
    # error status is passed to this method
    def on_error(self, status):
        print(status)
        # continue listening by "return True"
        return True


# In[5]:


# initiate an instance of MyListener 
tweet_listener=MyListener(output_file="python.csv",time_limit=10)

# start a staeam instance using authentication and the listener
twitter_stream = Stream(auth, tweet_listener)
# filtering tweets by topics
twitter_stream.filter(track=['#google', '#bofa','#bankofamerica','#netflix','#fb','#facebook','#apple','#microsoft','#boing','#amazon','#amazon','#jpmorgan',"#yahoo"])


# In[26]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#google'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[27]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_google.csv')






# In[28]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#bankofamerica'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[29]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_bankofamerica.csv')


# In[30]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#bofa'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[31]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_bofa.csv')


# In[35]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#netflix'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[36]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_netflix.csv')


# In[37]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#fb'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[38]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_fb.csv')


# In[39]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#apple'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[40]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_apple.csv')


# In[41]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#microsoft'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[42]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_microsoft.csv')


# In[43]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#boing'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[44]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_boing.csv')


# In[45]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#amazon'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[46]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_amazon.csv')


# In[47]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#jpmorgan'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[48]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_jpmorgan.csv')


# In[49]:


searched_tweets = []
tweets=[]
max_tweets=1000
last_id = -1

query='#yahoo'

api = tweepy.API(auth)


while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        # for each search, at maximum you get 100 results, although
        # you can set count larger than 100
        # You can limit the id for the most recent tweets (max_id)
        # query can be a list of hashtags
        # search api returns tweets sorted by time in descending order
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))

        if not new_tweets:
            break
        # append new batch into list    
        searched_tweets.extend(new_tweets)
        # only store a list of (date, tweet text) 
        tweets+=[(item.created_at, item.text) for item in new_tweets]
        
        # get the first tweet in the batch
        last_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# In[50]:


import pandas as pd
my_df = pd.DataFrame(tweets)
#print(my_df)
my_df.to_csv('python_yahoo.csv')

