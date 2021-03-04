#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from gensim.models import LdaModel
import pandas as pd
import ecopy as ep
import seaborn as sns
from scipy import stats


# In[2]:


nTopics = 45


# In[3]:


topic_names = [
'Water Quality',
'Sediment Transport',
'Wastewater Treatment',
'Flood Risk & Assessment',
'Hydrogeology',
'Coastal Hydrology', 
'River Flow',
'Wetland & Ecology',
'Runoff Quality',
'Rainfall-Runoff',
'Urban Drainage',
'Systems Hydrology',
'Surface-GW Interactions',
'Irrigation Water Management',
'Drought & Water Scarcity',
'Climate Change Impacts',
'Gauging & Monitoring',
'Forecasting',
'Glaciology',
'Salinity',
'Peatlands Mapping & Monitoring',
'Spatial Variability',
'Land Surface Flux',
'Solute Transport',
'Water Resources Management',
'Numerical Modeling',
'Hydrochemistry',
'Pollutant Removal',
'Groundwater Recharge',
'Uncertainty',
'Land Cover',
'Modeling & Calibration',
'Soil Moisture',
'Water Storage & Budgeting',
'Aquifers & Abstraction',
'Microbiology',
'Streamflow',
'Erosion',
'Dynamic Processes',
'Temporal Variability',
'Spatial Variability of Precipitation',
'Rainfall Intensity & Measurement',
'Watershed Hydrology',
'Hydraulics',
'Quantitative Analysis',
]


# In[4]:


topic_names_short = [
'WQ',
'SDT',
'WT',
'FRA',
'HG',
'CH', 
'RF',
'WE',
'RQ',
'RR',
'UD',
'SH',
'SGW',
'IWM',
'DWS',
'CC',
'GM',
'FC',
'GL',
'SN',
'PM',
'SV',
'LSF',
'SLT',
'WRM',
'NM',
'HC',
'PR',
'GWR',
'UC',
'LC',
'MDC',
'SM',
'WSB',
'AA',
'MCB',
'SF',
'ER',
'DP',
'TV',
'SVP',
'RIM',
'WH',
'HDR',
'QA',
]


# In[5]:


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
 'goldenrod':'#DAA520',
 'lawngreen':'#7CFC00',
 'rosybrown':'#BC8F8F',
 'mediumslateblue':'#7B68EE',
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
 'yellowgreen': '#9ACD32',
 'mistyrose': '#FFE4E1',
 'royalblue': '#4169E1',
 'lavender':  '#E6E6FA',
 'seashell': '#FFF5EE',
 'coral':'#FF7F50',
 'magenta':'#FF00FF',
 'moccasin':'#FFE4B5',
 'navy':'#000080',
 'paleturquoise':'#AFEEEE',
 'aliceblue':'#F0F8FF',
 'azure':'#F0FFFF',
 'khaki':'#F0E68C',
 'lightseagreen':'#20B2AA',
 'linen':'#FAF0E6',
 'palevioletred':'#DB7093',
 'sienna':'#A0522D',
 'mediumspringgreen':'#00FA9A',
 'indianred':'#CD5C5C',
 'tomato': '#FF6347',
}

# turn into a list
colorlist = []
for color in custom_colors.values():
    colorlist.append(tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))


# In[6]:


color_key_dict = {
'Water Quality':'burlywood',
'Sediment Transport':'chocolate',
'Wastewater Treatment':'crimson',
'Flood Risk & Assessment':'darkgreen',
'Hydrogeology':'darkorange',
'Coastal Hydrology':'darkslategrey', 
'River Flow':'deepskyblue',
'Wetland & Ecology':'dimgray',
'Runoff Quality':'firebrick',
'Rainfall-Runoff':'gold',
'Urban Drainage':'goldenrod',
'Systems Hydrology':'lawngreen',
'Surface-GW Interactions':'rosybrown',
'Irrigation Water Management':'mediumslateblue',
'Drought & Water Scarcity':'mediumvioletred',
'Climate Change Impacts':'orangered',
'Gauging & Monitoring':'orchid',
'Forecasting':'royalblue',
'Glaciology':'slateblue',
'Salinity':'springgreen',
'Peatlands Mapping & Monitoring':'steelblue',
'Spatial Variability':'teal',
'Land Surface Flux':'turquoise',
'Solute Transport':'yellow',
'Water Resources Management':'blueviolet',
'Numerical Modeling':'yellowgreen',
'Hydrochemistry':'mistyrose',
'Pollutant Removal':'royalblue',
'Groundwater Recharge':'lavender',
'Uncertainty':'seashell',
'Land Cover':'coral',
'Modeling & Calibration':'magenta',
'Soil Moisture':'moccasin',
'Water Storage & Budgeting':'navy',
'Aquifers & Abstraction':'paleturquoise',
'Microbiology':'aliceblue',
'Streamflow':'azure',
'Erosion':'khaki',
'Dynamic Processes':'lightseagreen',
'Temporal Variability':'linen',
'Spatial Variability of Precipitation':'palevioletred',
'Rainfall Intensity & Measurement':'sienna',
'Watershed Hydrology':'mediumspringgreen',
'Hydraulics':'indianred',
'Quantitative Analysis':'tomato',
}


# In[7]:


# Load model

lda_model = LdaModel.load(f'trained_models/trained_lda_model_new_{nTopics}')



# Load topic distributions

topic_distributions = np.load(f'data/topic_distributions_broad_{lda_model.num_topics}.npy')



# Pull topics

topics = lda_model.show_topics(formatted=False, num_topics=nTopics, num_words=20)



# load raw corpus dataframe

with open('data/raw_corpus_broad.pkl', 'rb') as f:
    corpus_df = pkl.load(f)


# In[8]:


# Pull journals

journals = corpus_df.Journal.unique()



# Pull years

years = np.unique(corpus_df['Year'])


# In[9]:


# Create a dictionary of topic distributions by year
#topic_distributions_year['Year'][paper][topic_weights]

topic_distributions_year = {}

for y, year in enumerate(years):
    
    topic_distributions_year[year] = topic_distributions[corpus_df['Year'] == year]


# In[10]:


topic_distributions_year['2011'].shape


# In[11]:


# Create a dictionary of topic distributions by journal
# topic_distributions_journal['Journal'][paper][topic_weights]

topic_distributions_journal = {}

for j, journal in enumerate(journals):
    
    topic_distributions_journal[journal] = topic_distributions[corpus_df['Journal'] == journal]
    


# In[12]:


topic_distributions_journal['HESS'][:,44]


# In[13]:


# Create a dictionary of topic distributions by year & journal
# topic_distributions_journal_year['Journal']['Year]'[paper][topic_weights]

topic_distributions_journal_year = {}

for j, journal in enumerate(journals):
    
    topic_distributions_journal_year[journal] = {}
    
    for y, year in enumerate(years):
        
        topic_distributions_journal_year[journal][year] = topic_distributions[(corpus_df['Journal'] == journal) & (corpus_df['Year'] == year)]


# In[14]:


# Create a dictionary of topic distributions by individual to


# In[15]:


# Define a list of diversity metrics for ecopy

diversity_metrics = ['shannon', 'spRich', 'gini-simpson', 'dominance']


# In[16]:


# Global diversity metrics
#global_diversity['metric'][paper_diversities]


global_diversity = {}

for metric in diversity_metrics:
    
    global_diversity[metric] = ep.diversity(topic_distributions, method = metric, breakNA=False, num_equiv=False)


# In[17]:


# Yearwise diversity metric
# year_diversity['metric']['year'][paper_diversities]

year_diversity = {}


for metric in diversity_metrics:
    
    year_diversity[metric] = {}
    
    for y, year in enumerate(years):
        
        year_diversity[metric][year] = ep.diversity(topic_distributions_year[year], method = metric, breakNA=False, num_equiv=False)


# In[18]:


# Journalwise diversity metric
# year_diversity['metric']['journal'][paper_diversities]

journal_diversity = {}

for metric in diversity_metrics:
    
    journal_diversity[metric] = {}
    
    for j, journal in enumerate(journals):
        
        journal_diversity[metric][journal] = ep.diversity(topic_distributions_journal[journal], method = metric, breakNA=False, num_equiv=False)


# In[19]:


#creating proxy artist for legends        
labels = list(color_key_dict.keys())
handles = [plt.Rectangle((0,0),1,1, color=color_key_dict[label]) for label in labels]
plt.axis('off')
plt.legend(handles, labels, shadow = True, fancybox = True, prop={'size': 20})
plt.savefig('figures/legends.png')  


# In[20]:


#Calculate r values per metric and plotting them.

r_vals_global = {}


for metric in diversity_metrics:
    
    r_vals_global[metric] = {}
    
    for t, topic in enumerate(topic_names):

        r_vals_global[metric][topic] = np.corrcoef(topic_distributions[:,t],global_diversity[metric])[0,1]

        
# Make the graph 20 inches by 40 inches
fig = plt.figure(figsize=(250,150), facecolor='white')


# plot numbering starts at 1, not 0
plot_number = 1

for metric in diversity_metrics:
    ax = fig.add_subplot(2, 2, plot_number)
    ax.bar(topic_names_short, r_vals_global[metric].values(), color = list(custom_colors.values()))
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=100)
    ax.set_title(metric, size = 150)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()
plt.savefig('figures/r_allmetrics_fullcorpus.png')    


# ### statistical relationship between paper-topic distributions and diversities by journal

# In[21]:


#Create a dictionary of [metric][journal][topic][weights]
journal_diversity_correlation_dict = {}


    
for metric in diversity_metrics:

    journal_diversity_correlation_dict[metric] = {}

    for j, journal in enumerate(journals):

        journal_diversity_correlation_dict[metric][journal] = {}

        
        for t, topic in enumerate(topic_names):

            journal_diversity_correlation_dict[metric][journal][topic] = np.corrcoef(topic_distributions_journal[journal][:,t],journal_diversity[metric][journal])[0,1]
            
            


# In[22]:


# Make the graph 20 inches by 40 inches
fig = plt.figure(figsize=(250,150), facecolor='white')


# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = fig.add_subplot(6, 3, plot_number)
    ax.bar(topic_names_short, journal_diversity_correlation_dict['shannon'][journal].values(), color = list(custom_colors.values()))
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=100)
    ax.set_title(journal, size = 150)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()
plt.savefig('figures/r_shannon_alljournals.png')    


# In[37]:


fig = plt.figure(figsize=(250,150), facecolor='white')


# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = fig.add_subplot(6, 3, plot_number)
    ax.bar(topic_names_short, journal_diversity_correlation_dict['gini-simpson'][journal].values(), color = list(custom_colors.values()))
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=100)
    ax.set_title(journal, size = 150)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()
plt.savefig('figures/r_gini_alljournals.png')    


# In[38]:


fig = plt.figure(figsize=(250,150), facecolor='white')


# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = fig.add_subplot(6, 3, plot_number)
    ax.bar(topic_names_short, journal_diversity_correlation_dict['spRich'][journal].values(), color = list(custom_colors.values()))
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=100)
    ax.set_title(journal, size = 150)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()
plt.savefig('figures/r_sprich_alljournals.png')    


# In[39]:


fig = plt.figure(figsize=(250,150), facecolor='white')


# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = fig.add_subplot(6, 3, plot_number)
    ax.bar(topic_names_short, journal_diversity_correlation_dict['dominance'][journal].values(), color = list(custom_colors.values()))
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=100)
    ax.set_title(journal, size = 150)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()
plt.savefig('figures/r_dominance_alljournals.png')    


# ### statistical relationship between paper-topic distributions and diversities by year ###

# In[23]:


#Creating topicwise distribution dictionary
#topicwise_dist_dict[paper][topic]

topicwise_year_dict = {}

for y, year in enumerate(years):
    
    topicwise_year_dict[year] = {}

    for p in range(len(topic_distributions_year[year])):

        topicwise_year_dict[year][p] = {}

        for t, topic in enumerate(topic_names):

            topicwise_year_dict[year][p][topic] = topic_distributions_year[year][p][t]


# In[24]:


topicwise_year_dict['2011'][1]


# In[25]:


#dictionary of dataframes for individual years
df_year_weights_dict = {}

for y, year in enumerate(years):
    df_year_weights_dict[year] = (pd.DataFrame(topicwise_year_dict[year])).transpose()

df_year_weights_dict['2011']['Climate Change Impacts']


# In[26]:


#Use this dictionary to iterate over year and topic names for r-sq calculations

year_topicwise_dict = {}

for y, year in enumerate(years):
    year_topicwise_dict[year] = {}
    for t, topic in enumerate(topic_names):
        year_topicwise_dict[year][topic] = np.array(df_year_weights_dict[year][topic])

year_topicwise_dict['2011']


# In[27]:


#year_diversity_correlation_dict[metric][year][topic][rsq]
year_diversity_correlation_dict = {}


for metric in diversity_metrics:

    year_diversity_correlation_dict[metric] = {}

    for y, year in enumerate(years):

        year_diversity_correlation_dict[metric][year] = {}

        for t, topic in enumerate(topic_names):

            year_diversity_correlation_dict[metric][year][topic] = np.corrcoef(year_topicwise_dict[year][topic],year_diversity[metric][year])[0,1]


# In[28]:


year_diversity_correlation_dict['shannon']['2011'].values()


# In[29]:


plt.figure(figsize=(55,35))
plt.xticks(size = 30, rotation = 75)
plt.yticks(size = 30)
plt.ylabel('R-Squared', size = 35)
plt.title('Statistical Relationship Between Paper-Topic Distributions and Topic Dominance for 2011', size = 45)
plt.bar(year_diversity_correlation_dict['shannon']['2011'].keys(),year_diversity_correlation_dict['shannon']['2011'].values(), color = 'purple')
 


# In[30]:


#shared x and y
#For journals gini
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = plt.subplot(6, 3, plot_number)
    ax.bar(topic_names, journal_diversity_correlation_dict['gini-simpson'][journal].values())
    ax.set_title(journal, size = 40)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()


# In[31]:


#For journals sprich
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = plt.subplot(6, 3, plot_number)
    ax.bar(topic_names, journal_diversity_correlation_dict['spRich'][journal].values())
    ax.set_title(journal, size = 40)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()


# In[32]:


#For journals dominance
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for journal in journals:
    ax = plt.subplot(6, 3, plot_number)
    ax.bar(topic_names, journal_diversity_correlation_dict['dominance'][journal].values())
    ax.set_title(journal, size = 40)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1

plt.tight_layout()


# In[33]:


# for years shannon
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for year in years:
    ax = plt.subplot(5, 6, plot_number)
    ax.bar(topic_names, year_diversity_correlation_dict['shannon'][year].values())
    ax.set_title(year, size = 30)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1


plt.tight_layout()


# In[34]:


# for years gini
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for year in years:
    ax = plt.subplot(5, 6, plot_number)
    ax.bar(topic_names, year_diversity_correlation_dict['gini-simpson'][year].values())
    ax.set_title(year, size = 30)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1


plt.tight_layout()


# In[35]:


# for years sprich
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for year in years:
    ax = plt.subplot(5, 6, plot_number)
    ax.bar(topic_names, year_diversity_correlation_dict['spRich'][year].values())
    ax.set_title(year, size = 30)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1


plt.tight_layout()


# In[36]:


# for years dominance
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(40,80), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1

for year in years:
    ax = plt.subplot(5, 6, plot_number)
    ax.bar(topic_names, year_diversity_correlation_dict['dominance'][year].values())
    ax.set_title(year, size = 30)
    # Go to the next plot for the next loop
    plot_number = plot_number + 1


plt.tight_layout()


# In[ ]:




