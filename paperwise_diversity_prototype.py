#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from gensim.models import LdaModel
import pandas as pd
import ecopy as ep



nTopics = 45

# Load model
lda_model = LdaModel.load(f'trained_models/trained_lda_model_new_{nTopics}')
# Load topic distributions
topic_distributions = np.load(f'data/topic_distributions_broad_{lda_model.num_topics}.npy')
topic_distributions_wogw = np.load(f'data/topic_distributions_broad_wogw_{lda_model.num_topics}.npy')
# Pull topics
topics = lda_model.show_topics(formatted=False, num_topics=nTopics, num_words=20)
# load raw corpus dataframe
with open('data/raw_corpus_broad.pkl', 'rb') as f:
    corpus_df = pkl.load(f)
with open('data/raw_corpus_broad_wogw.pkl', 'rb') as f:
    corpus_df_wogw = pkl.load(f)


# In[ ]:


# Pull years
years = np.unique(corpus_df['Year'])


# In[ ]:


# Pull papers
papers = np.unique(corpus_df['Title'])


# In[ ]:


# Paper wise diversity metrics
diversity_metrics = ['shannon', 'simpson', 'gini-simpson', 'dominance', 'even']

shannon_diversity_paper = {}
simpson_diversity_paper = {}
gini_diversity_paper = {}
dominance_paper = {}
shannon_diversity_mean_paper = []
simpson_diversity_mean_paper = []
gini_diversity_mean_paper = []
dominance_mean_paper = []

arr = np.full([len(years),nTopics], np.nan)
                           
for y, year in enumerate(years):
    for p, paper in enumerate(papers):
        #making an array of year-paper wise topic distributions 
        topic_distributions_paper = topic_distributions[(corpus_df['Year'] == year) & (corpus_df['Title'] == paper),:]
        np.append(topic_distributions_paper,arr)

with open('topic_distribution_paperwise.npy', 'wb') as f:
    np.save(f, topic_distributions_paper)


# In[ ]:


for y, year in enumerate(years):
    shannon_diversity_paper[year] = ep.diversity(topic_distributions_paper, method = 'shannon', breakNA=False, num_equiv=False)
    simpson_diversity_paper[year] = ep.diversity(topic_distributions_paper, method = 'simpson', breakNA=False, num_equiv=False)
    gini_diversity_paper[year] = ep.diversity(topic_distributions_paper, method = 'gini_simpson', breakNA=False, num_equiv=False)
    dominance[year] = ep.diversity(topic_distributions_paper, method = 'dominance', breakNA=False, num_equiv=False)
    
    
    shannon_diversity_mean_paper.append(np.mean(shannon_diversity_paper[year]))
    simpson_diversity_mean_paper.append(np.mean(simpson_diversity_paper[year]))
    gini_diversity_mean_paper.append(np.mean(gini_diversity_paper[year]))
    dominance_mean_paper.append(np.mean(dominance_paper[year]))


fig, axs = plt.subplots(2,2,figsize=(15,15))    
axs[0, 0].plot(shannon_diversity_mean_paper[:-1])
axs[0, 0].set_title('Shannon (paperwise)')
axs[0, 1].plot(simpson_diversity_mean_paper[:-1], 'tab:orange')
axs[0, 1].set_title('Simpson')
axs[1, 0].plot(gini_diversity_mean_paper[:-1],'tab:green')
axs[1, 0].set_title('Gini-Simpson')
axs[1, 1].plot(dominance_mean_paper[:-1], 'tab:red')
axs[1, 1].set_title('Dominance')
plt.savefig('figures/diversity_paper_year_mean.png')
    


# In[ ]:




