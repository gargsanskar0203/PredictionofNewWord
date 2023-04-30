#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import tensorflow as tf
import numpy as np


# In[4]:


df = pd.read_csv('tmdb_5000_credits.csv')
df.head(5)


# In[6]:


df = df['title']


# In[7]:


df


# In[8]:


movie_name = df.to_list()


# In[9]:


movie_name


# In[10]:


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(movie_name)
seq = tokenizer.texts_to_sequences(movie_name)


# In[11]:


seq[:10]


# In[12]:


tokenizer.word_index


# In[13]:


X = []
y = []
total_words_dropped = 0

for i in seq:
    if len(i) > 1:
        for index in range(1, len(i)):
            X.append(i[:index])
            y.append(i[index])
    else:
        total_words_dropped += 1

print("Total Single Words Dropped are:", total_words_dropped)


# In[14]:


X[:10]


# In[15]:


y[:10]


# In[16]:


X = tf.keras.preprocessing.sequence.pad_sequences(X)


# In[17]:


X


# In[18]:


X.shape


# In[19]:


y = tf.keras.utils.to_categorical(y)


# In[20]:


y


# In[21]:


y.shape


# In[22]:


vocab_size = len(tokenizer.word_index) + 1


# In[23]:


vocab_size


# In[24]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 14),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax'),
])


# In[25]:


model.summary()


# In[26]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.004), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])


# In[27]:


model.fit(X, y, epochs=150)


# In[29]:


model.save('nwp.h5')


# In[30]:


vocab_array = np.array(list(tokenizer.word_index.keys()))


# In[31]:


vocab_array


# In[32]:


def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text


# In[33]:


make_prediction("cloudy", 5)


# In[ ]:




