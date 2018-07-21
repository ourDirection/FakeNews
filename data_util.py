
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np

import spacy
import nltk
import re

import random
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

tqdm.pandas(desc='progress-bar')


# In[61]:


stop = set(stopwords.words('english') + list(string.punctuation))
stemmer = PorterStemmer()
re_punct = re.compile('[' + ''.join(string.punctuation) + ']')

def preprocess(text):
    try:
        text = text.lower()
        tokens = word_tokenize(text)
#         tokens = [t for t in tokens if not t in stop]
        tokens = [re.sub(re_punct, ' ', t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]
#         tokens = [stemmer.stem(t) for t in tokens]
        if len(tokens) == 0:
            return None
        else:
            return ' '.join(tokens)
    except:
        return None
    
def tokenize_count(row):
    return len(row.split(' '))



# In[16]:


def prep_fake_or_real_news():
    data_train = pd.read_csv('data/fake_or_real_news.csv')
    data_train['tokens'] = data_train['text'].progress_map(preprocess)
    
    data_train = data_train[data_train['tokens'].notnull()]
    data_train.reset_index(inplace=True)
    data_train.drop('index', inplace=True, axis=1)
    data_train.to_csv('data/fake_or_real_news.csv')

# prep_fake_or_real_news()


# In[17]:


def get_splits(X, y, max_words = 20000, MAX_SEQUENCE_LENGTH = 100, EMBEDDING_DIM = 300):
    tokenizer = Tokenizer(num_words=max_words,lower=True, split=' ', 
                          filters='"#%&()*+-/<=>@[\\]^_`{|}~\t\n',
                          char_level=False, oov_token=u'<UNK>')

    tokenizer.fit_on_texts(X)

    X = tokenizer.texts_to_sequences(X)
    Y = pd.get_dummies(y).values
    # print(X[0])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH,  padding="post", truncating="post")
    # print(X[10])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, 
                                                        random_state = 42)

    # print(y_train[100])

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    
    return x_train, x_test, y_train, y_test, word_index


# In[24]:


def get_fake_or_real_news():
    data_train = pd.read_csv('data/fake_or_real_news.csv')
    data_train['label'].value_counts().plot(kind='bar', alpha=.5)

    print('Loading data...')

    X = data_train['tokens'].values
    y = data_train['label']

    classes = data_train['label'].unique()
    
    labels = data_train['label'].unique()
    print(labels)
    
    num_classes = len(labels)
    print(num_classes, 'classes')
    
    x_train, x_test, y_train, y_test, word_index = get_splits(X, y)
    
    return x_train, x_test, y_train, y_test, word_index, labels, num_classes, X, y


# x_train, x_test, y_train, y_test, word_index, labels, num_classes, X, y = get_fake_or_real_news()


# In[22]:


def get_politifact():
    data_train = pd.read_csv('/home/paperspace/sonic/fakeNews/data/politifact.tsv', sep='\t')

    # data_train['ruling'].value_counts().plot(kind='bar', alpha=.5)

    data_train['label'] = '' 
    def label(row):
    #     print(row)
        if row in ['true', 'half-true', 'mostly-true']:
            return 'true'
        else:
            return 'false'

    data_train['label'] = data_train['ruling'].apply(label)

    print('Loading data...')
    
        ###############
    X  = data_train['statement__text'].values
    y = data_train['label']

    classes = data_train['label'].unique()
    
    labels = data_train['label'].unique()
    print(labels)
    
    num_classes = len(labels)
    print(num_classes, 'classes')
    
    x_train, x_test, y_train, y_test, word_index = get_splits(X, y)

    data_train['label'].value_counts().plot(kind='bar', alpha=.5)
    
    return x_train, x_test, y_train, y_test, word_index, labels, num_classes, X, y
 
# x_train, x_test, y_train, y_test, word_index, labels, num_classes = get_politifact()


# In[69]:



def prep_UCI():
    header = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME','TIMESTAMP']
    # b = business, t = science and technology, e = entertainment, m = health
    data_train = pd.read_csv('data/newsCorpora.csv', names=header,  sep='\t')
    
    data_train['TITLE'] = data_train['TITLE'].progress_map(preprocess)

    data_train = data_train[data_train['TITLE'].notnull()]
    data_train.drop('ID', inplace=True, axis=1)
    data_train.reset_index(inplace=True)
    data_train['token_count'] = data_train['TITLE'].apply(tokenize_count)
    data_train = data_train[data_train['token_count'] < 50]
    
    return data_train
    
# data_train = prep_UCI()
# data_train['CATEGORY'].value_counts().plot(kind='bar', alpha=.5)
# data_train['token_count'].hist(bins=10, alpha=.5)
# # data_train.to_csv('data/newsCorpora.csv', sep='\t')
# data_train.tail()
    


# In[76]:



# data_train['token_count'].hist(bins=10, alpha=.5)
# data_train['token_count'].describe()


# In[96]:



def prep_fake():
    data_train = pd.read_csv('data/fake.csv')
    data_train['text'] = data_train['text'].progress_map(preprocess)
    data_train['thread_title'] = data_train['thread_title'].progress_map(preprocess)

    data_train = data_train[data_train['thread_title'].notnull()]
    data_train.reset_index(inplace=True)
    data_train.drop('index', inplace=True, axis=1)
    data_train['token_count'] = data_train['thread_title'].apply(tokenize_count)
    return data_train

# fake_df = prep_fake()
# fake_df.to_csv('data/fake.csv')
# fake_df.tail()


# In[78]:


# fake_df['token_count'].hist(bins=10, alpha=.5)
# fake_df['token_count'].describe()


# In[95]:





# In[98]:


def get_fact_fake():
    # print('Loading data...')

    fake_df = pd.read_csv('data/fake.csv')
    real_df = pd.read_csv('data/newsCorpora.csv', sep='\t')
    text = np.append(fake_df['thread_title'].values,  real_df['TITLE'].values)
    label = np.append( ['FALSE'] * len(fake_df), ['TRUE'] * len(real_df))
    data = {'text' : text, 'label': label}

    data_train = pd.DataFrame(data)
    data_train['label'].value_counts().plot(kind='bar', alpha=.5)
    print(data_train['label'].value_counts())
    data_train.tail()

    X  = data_train['text'].values
    y = data_train['label']

    labels = data_train['label'].unique()
    print(labels)

    num_classes = len(labels)
    print(num_classes, 'classes')

    x_train, x_test, y_train, y_test, word_index = get_splits(X, y)

#     data_train['label'].value_counts().plot(kind='bar', alpha=.5)

    return x_train, x_test, y_train, y_test, word_index, labels, num_classes, X, y

# x_train, x_test, y_train, y_test, word_index, labels, num_classes, _ = get_fact_fake()
