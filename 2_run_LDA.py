#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pickle as pkl
import gensim
import numpy as np
from tqdm.notebook import tqdm
from gensim.models.ldamulticore import LdaMulticore

# In[2]:


# enable logging
logging.basicConfig(filename='log_files/gensim_prespecified_topics.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)


# In[3]:


# load cleaned corpus
with open('data/cleaned_corpus.pkl', 'rb') as f:
    corpus = pkl.load(f)
with open("data/id2word.pkl", 'rb') as f:
    id2word= pkl.load(f)


# In[4]:


# Choose the number of topics
nTopics = 17


# In[5]:


# Train the LDA model with a prespecified number of topics
lda_model =                   LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=nTopics, 
                                           random_state=100,
                                           chunksize=200,
                                           passes=2000,
#                                            iterations=10000,
#                                            minimum_probability=0,
                                           per_word_topics=True)


# In[6]:


# Save the trained LDA model
lda_model.save(f"trained_models/trained_lda_model_{lda_model.num_topics}")


# In[7]:


# Run the model
doc_lda = lda_model[corpus]


# In[8]:


# Extract the topic distributions for each paper as numpy array
hm = np.zeros([len(corpus), lda_model.num_topics])
for i in tqdm(range(len(doc_lda))):
    for topic_pair in doc_lda[i][0]:
        hm[i, topic_pair[0]] = topic_pair[1]


# In[9]:


# Save topic distributions as numpy array
np.save(f'data/topic_distributions_{lda_model.num_topics}', hm)

