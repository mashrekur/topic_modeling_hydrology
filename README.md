# topic_modeling_hydrology

--- Initial Setup --------------------

1) Change the prefix in the environment.yml file to point to your conda environemnt folder on your local machine.
2) Create the conda environment: `conda env create -f environment.yml` 

3) Download the nltk stopwords file from <https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip> and put it in <~/anaconda3/envs/nlp/nltk_data/corpora> or in one of the other paths that nltk checks.

4) Download spacy data as: `python -m spacy download en`

--- Run the Training Procedure -------

0_data_preprocessing.ipynb --- Cleans and prepares the corpus. This produces four output files in the <data> directory: 
(i) raw_corpus.pkl is the pandas dataframe containing the whole abstract corpus 
(ii) data_lemmatized.pkl is the lemmatized data 
(iii) id2word.pkl is the word <-> id mapping
(iv) cleaned_corpus.pkl is the term document frequency matrix

1_optimal_topics.py --- Runs a loop to train an LDA model over variable number of topics to find the best number of topics to use for this corpus. Can be run using the slurm script ` 

2_run_LDA 

(The last notebook does not need to be run, but if you skip this step then it is necessary to change the load statements throught the subsequent scripts.) 

--- Figures & Analysis ----------------





