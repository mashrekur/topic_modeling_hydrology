#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle as pkl
import pandas as pd 
from gensim.models import LdaModel
import json
import pickle


# In[2]:


nTopics = 30


# In[3]:


# Load model
lda_model = LdaModel.load(f'../trained_models/trained_lda_model_new_{nTopics}')

# Load topic distributions
topic_distributions = np.load(f'../data/topic_distributions_new_{lda_model.num_topics}.npy')

# Pull topics
topics = lda_model.show_topics(formatted=False, num_topics=nTopics, num_words=20)

# load raw corpus dataframe
with open('../data/raw_corpus_mallet.pkl', 'rb') as f:
    corpus_df = pkl.load(f)


# In[4]:


#convert all nans to zeros and all zeros to a very small number
# topic_distributions = np.nan_to_num(topic_distributions)
topic_distributions = np.where(topic_distributions == 0, 0.000001, topic_distributions)


# In[ ]:


#calculate JSD for all pairs of papers
#the max force values (dist) are capped to 1000 later on
def calc_KL_divergence(paper1,paper2):
    return -np.nansum(paper1 * np.log(paper2/paper1))
def jensen_shannon_distance(paper1,paper2):
    M=0.5*(paper1+paper2)
    D1=calc_KL_divergence(paper1,M)
    D2=calc_KL_divergence(paper2,M)
    JSDiv = 0.5*D1+0.5*D2
    JSD = np.sqrt(JSDiv)
    return JSD


# In[ ]:


jsd_arr  = {}

topic_list = list(range(30))

for k1, t1 in enumerate(topic_list):
    for k2, t2 in enumerate(topic_list):
        fname = str(t1).zfill(2)+str(t2).zfill(2)
        if fname in jsd_arr.keys():
            continue
        jsd_arr[fname] = np.array([])
        if k1==k2:
            calc_jsd = False
        else:
            calc_jsd = True
            for p1, paper1 in enumerate(corpus_df["Title"][:]):
                for p2, paper2 in enumerate(corpus_df["Title"][:]):

                    if p1 == p2:
                        dist = 0
                    else:
                        JSD = jensen_shannon_distance(topic_distributions[p1, k1], topic_distributions[p2, k2])
                        dist = round(1/JSD, 2)
                        jsd_arr[fname] = np.append(jsd_arr[fname], dist)
#             print(jsd_arr[fname])
            np.save(fname, jsd_arr[fname], allow_pickle=True, fix_imports=True)
             

