#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle as pkl
import pandas as pd 
from gensim.models import LdaModel
import json


# In[ ]:


nTopics = 25


# In[ ]:


# Load model
lda_model = LdaModel.load(f'trained_models/trained_lda_model_{nTopics}')

# Load topic distributions
topic_distributions = np.load(f'data/topic_distributions_{lda_model.num_topics}.npy')

# Pull topics
topics = lda_model.show_topics(formatted=False, num_topics=nTopics, num_words=20)

# load raw corpus dataframe
with open('data/raw_corpus.pkl', 'rb') as f:
    corpus_df = pkl.load(f)


# In[ ]:


topic_distributions.shape


# In[ ]:


# Define topic names
topic_names = [
    'Precip Variability & Extr',
    'Hydrogeochemistry',
    'Uncertainty',
    'Soil Moisture',
    'Statistical Hydrology',
    'Rainfall-Runoff',
    'Precip Observation',
    'Modeling & Calibration',
    'Water Management',
    'Snow Hydrology',
    'Streamflow Processes',
    'Water Quality',
    'Channel Flow',
    'Floods',
    'Sediment & Erosion',
    'Climate Change',
    'Subsurface Flow & Trans',
    'Scaling & Spatial Variabil',
    'Land Surface Fluxes',
    'Hydrogeology',
    'Human Interv & Eff',
    'Land Cover',
    'Systems Hydrology',
    'Modeling & Forecasting',
    'Groundwater'
]


# In[ ]:


# Define colors to associate with each topic
custom_colors = {
 'burlywood': '#DEB887',
 'chocolate': '#D2691E',
 'crimson': '#DC143C',
 'darkgreen': '#006400',
 'darkorange': '#FF8C00',
 'darkslategrey': '#2F4F4F',
 'deepskyblue': '#00BFFF',
 'dimgray': '#696969',
 'firebrick': '#B22222',
 'gold': '#FFD700',
 'goldenrod': '#DAA520',
 'lawngreen': '#7CFC00',
 'lightcoral': '#F08080',
 'lightpink': '#FFB6C1',
 'mediumvioletred': '#C71585',
 'orangered': '#FF4500',
 'orchid': '#DA70D6',
 'royalblue': '#4169E1',
 'slateblue': '#6A5ACD',
 'springgreen': '#00FF7F',
 'steelblue': '#4682B4',
 'teal': '#008080',
 'turquoise': '#40E0D0',
 'yellow': '#FFFF00',
 'blueviolet': '#8A2BE2',
 'yellowgreen': '#9ACD32'}

# turn into a list
colorlist = []
for i, color in enumerate(custom_colors.values()):
    colorlist.append(tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    colorlist[i] = (colorlist[i][0] / 256, colorlist[i][1] / 256, colorlist[i][2] / 256)


# In[ ]:


#calculate JSD for all pairs of papers
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


# jensen_shannon_distance(0.12, 0.74)


# In[ ]:


#initiate individual lists for nodes and links
node_list = []
link_list = []


for p1, paper1 in enumerate(corpus_df["Title"]):
    grp = {"group" : np.argmax(topic_distributions[p1]), "name": paper1}
    node_list.append(grp)
    for p2, paper2 in enumerate(corpus_df["Title"]):
        if p1 == p2:
            dist = 0
        else:
            #round to 2 decimal places and multiply by 10
            dist = int(round(1/jensen_shannon_distance(topic_distributions[p1], topic_distributions[p2]), 2)*10)
        link = {"source": p1, "target": p2, "value": dist}
        print(link)
        link_list.append(link)


# In[ ]:


#initiate json file
json_prep = {"links":link_list, "nodes":node_list}

#json does not recognize NumPy data types; defining own encoder
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#dumping the data into json file
json_dump = json.dumps(json_prep, indent=1, sort_keys=True, cls=NpEncoder)


# In[ ]:


#pd.DataFrame(json_prep['nodes']).head()


# In[ ]:


# pd.DataFrame(json_prep['links']).head()


# In[ ]:


#save output
filename_out = 'hydro_mind.json'
json_out = open(filename_out,'w')
json_out.write(json_dump)
json_out.close()


# In[ ]:




