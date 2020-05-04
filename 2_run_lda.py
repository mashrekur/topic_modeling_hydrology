import logging
import pickle as pkl
import gensim
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
import argparse

def run_lda(num_topics):
    
    print('Number of Topics: ', num_topics)

    # enable logging
    logging.basicConfig(filename=f'log_files/gensim_prespecified_topics_{num_topics}.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

    # load cleaned corpus
    with open('data/cleaned_corpus.pkl', 'rb') as f:
        corpus = pkl.load(f)
    with open("data/id2word.pkl", 'rb') as f:
        id2word= pkl.load(f)

    # Train the LDA model with a prespecified number of topics
    lda_model =                   LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics, 
                                               random_state=100,
                                               chunksize=100,
                                               passes=3000,
    #                                            iterations=10000,
    #                                            minimum_probability=0,
                                               per_word_topics=True)


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
    
    return
    
if __name__ == "__main__":
    
    # input argumnets
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-topics', type=int)
    args = vars(parser.parse_args())
    
    # run the training procedure
    run_lda(args['num_topics'])