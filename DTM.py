#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pickle as pkl
import gensim
import numpy as np
from gensim.models import LdaSeqModel


# In[ ]:


# enable logging
logging.basicConfig(filename='log_files/DTM_prespecified_topics.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)


# In[ ]:


# load cleaned corpus
with open('data/raw_corpus_dtm.pkl', 'rb') as f:
    corpus_df = pkl.load(f)
with open('data/cleaned_corpus_dtm.pkl', 'rb') as f:
    corpus = pkl.load(f)
with open("data/id2word_dtm.pkl", 'rb') as f:
    id2word= pkl.load(f)


# In[ ]:


#takes all unique values in data - year as well as how often they occur and returns them as an array.
uniqueyears, time_slices = np.unique(corpus_df.Year, return_counts=True) 
#this array will be used for time slicing while training the LDA sequential model
print(np.asarray((uniqueyears, time_slices)).T) 


# In[ ]:


# Choose the number of topics
nTopics = 35


# In[ ]:


# Train the LDA model with a prespecified number of topics
lda_model =                   LdaSeqModel(corpus=corpus,
                                          time_slice=time_slices,
                                           id2word=id2word,
                                           num_topics=nTopics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=2000)


# In[ ]:


# Save the trained DTM
lda_model.save(f"trained_models/trained_DTM_{lda_model.num_topics}")


# In[ ]:


# Extract the topic distributions for each paper as numpy array
hm = np.zeros([len(corpus), lda_model.num_topics])
for i in range(len(doc_lda)):
    for topic_pair in doc_lda[i][0]:
        hm[i, topic_pair[0]] = topic_pair[1]


# In[ ]:


# Save topic distributions as numpy array
np.save(f'data/topic_distributions_DTM_{lda_model.num_topics}', hm)

