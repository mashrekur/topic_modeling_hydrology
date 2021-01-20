import pickle as pkl
import logging
import numpy as np
import gensim
import matplotlib.pyplot as plt
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

# Switch for running new experiemnts vs. loading old ones
NEW_EXPERIMENTS = True

# load cleaned corpus
with open('data/data_lemmatized_broad.pkl', 'rb') as f:
    data_lemmatized = pkl.load(f)
with open('data/cleaned_corpus_broad.pkl', 'rb') as f:
    corpus = pkl.load(f)
with open("data/id2word_broad.pkl", 'rb') as f:
    id2word= pkl.load(f)

# parameters of topic size search
#max_topics = 40
#min_topics = 5
#max_num_topics = 500
#try_ntopics = np.unique(np.logspace(np.log10(min_topics), np.log10(max_topics+1), max_num_topics).astype(int))
#num_ntopics = len(try_ntopics)

try_ntopics = np.array(list(range(10,81,5)))
num_ntopics = len(try_ntopics)

# save the list of numbers of topics to try
np.save("data/try_ntopics",try_ntopics)

# init storage
perplexity = {}
coherence = {}

# loop through topic sizes
if NEW_EXPERIMENTS:
    for i, topics in enumerate(try_ntopics):

        # enable logging
        logging.basicConfig(filename=f'log_files/gensim_{topics}_topics_logscale.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

        # Build LDA model with this number of topics
        lda_model =                LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=topics, 
                                               random_state=100,
                                               chunksize=200,
                                               passes=1000,
    #                                            iterations=5000,
    #                                            minimum_probability=0,
                                               per_word_topics=True)

        #Compute Perplexity
        perplexity[topics] = lda_model.log_perplexity(corpus)  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence[topics] = coherence_model_lda.get_coherence()

        #save results
        lda_model.save(f"trained_models/trained_lda_model_search_broad_{topics}")
        with open("data/perplexity.pkl", 'wb') as f:
            pkl.dump(perplexity, f)
        with open("data/coherence.pkl", 'wb') as f:
            pkl.dump(coherence, f)         

        # screen report
        print(f"Num Topics = {topics}: Perplexity = {perplexity[topics]}, Coherence = {coherence[topics]}")
        

# Load perplexity and coherence scores for plotting
with open("data/perplexity.pkl", 'rb') as f:
    perplexity = pkl.load(f)
with open("data/coherence.pkl", 'rb') as f:
    coherence = pkl.load(f)         

# Plot coherence and perplexity scores

# grab colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# init figure
fig, ax = plt.subplots(figsize=(12,6))

# plot perplexity
lists = sorted(perplexity.items())
x, y = zip(*lists) 
pltp = ax.plot(x, y, label='perplexity', linewidth=5) 

# plot coherence
ax2 = ax.twinx() 
lists = sorted(coherence.items())
x, y = zip(*lists) 
pltc = ax2.plot(x, y, label='coherence', linewidth=5, color = colors[2])

# axis labels
ax.set_xlabel('Number of Topics', fontsize=14)
ax.set_ylabel('Perplexity', fontsize=14)
ax2.set_ylabel('Coherence', fontsize=14)
ax.set_title('Finding the Optimal Number of Topics', fontsize=20)

# legend
ax.legend(pltp+pltc, ['Perplexity', 'Coherence'], fontsize=14, loc='best')

# aesthetics
ax.grid()

# Save figure
plt.savefig('figures/perplexity_coherence_broad.png')

