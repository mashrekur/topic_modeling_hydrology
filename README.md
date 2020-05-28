# topic_modeling_hydrology
DOI: <a href="https://zenodo.org/badge/latestdoi/254929556"><img src="https://zenodo.org/badge/254929556.svg" alt="DOI"></a>

--- Initial Setup --------------------

1) Change the prefix in the environment.yml file to point to your conda environemnt folder on your local machine.
2) Create the conda environment: `conda env create -f environment.yml` 

3) Download the nltk stopwords file from <https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip> and put it in <~/anaconda3/envs/nlp/nltk_data/corpora> or in one of the other paths that nltk checks.

4) Download spacy data as: `python -m spacy download en`

5) You might have to update holoviews `conda update holoviews`

--- Run the Training Procedure -------

0_data_preprocessing.ipynb --- Cleans and prepares the corpus. This produces four output files in the <data> directory: 
(i) raw_corpus.pkl is the pandas dataframe containing the whole abstract corpus 
(ii) data_lemmatized.pkl is the lemmatized data 
(iii) id2word.pkl is the word <-> id mapping
(iv) cleaned_corpus.pkl is the term document frequency matrix

1_optimal_topics.py --- Runs a loop to train an LDA model over variable number of topics to find the best number of topics to use for this corpus. Can be run using the slurm script ` 

2_run_LDA --- This is a larger run (more passes and smaller chunk size) for the chosen number of topics for final analysis.Also created the topic distribution matrix for this run. 

Note: The <2_run_LDA> script will extract teh per-paper topic distributions and store this as a numpy array in the <data> directory. You can also use the <X_extract_topic_distributions.ipynb> notebook to do this manually. It takes a few minutes per trained model.

--- Figures & Analysis ----------------

3_subjective_topic_analysis --- Tools and workspace to help understand and name topics 
4_temporal_topics --- Creates plots for analysis of relative temporal topic trends
5_topic_correlations --- Creates plots for analysis of topic relationships
6_journal_correlations --- Creates plots for analysis of journal relationships
7_journal_correlations --- Creates plots for analysis of how topics evolve as the number of topics increases 



