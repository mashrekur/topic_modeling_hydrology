import logging
import pickle as pkl
import gensim
import numpy as np
from tqdm.notebook import tqdm
from gensim.models.ldamulticore import LdaMulticore

# enable logging
logging.basicConfig(filename='log_files/gensim_prespecified_topics.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)

# load cleaned corpus
with open('data/cleaned_corpus.pkl', 'rb') as f:
    corpus = pkl.load(f)
with open("data/id2word.pkl", 'rb') as f:
    id2word= pkl.load(f)

# Choose the number of topics
nTopics = 30

# Train the LDA model with a prespecified number of topics
lda_model =                   LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=nTopics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=3000,
#                                            iterations=10000,
#                                            minimum_probability=0,
                                           per_word_topics=True)


# Save the trained LDA model
lda_model.save(f"trained_models/trained_lda_model_{lda_model.num_topics}")

# Run the model
doc_lda = lda_model[corpus]

# Extract the topic distributions for each paper as numpy array
hm = np.zeros([len(corpus), lda_model.num_topics])
for i in tqdm(range(len(doc_lda))):
    for topic_pair in doc_lda[i][0]:
        hm[i, topic_pair[0]] = topic_pair[1]

# Save topic distributions as numpy array
np.save(f'data/topic_distributions_{lda_model.num_topics}', hm)

