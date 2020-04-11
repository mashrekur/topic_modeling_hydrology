# topic_modeling_hydrology

--- Initial Setup --------------------

1) Need to create an environment YAML file --> Instructions here for the user to source the environment file.

2) Download the nltk stopwords file from <https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip> and put it in <~/anaconda3/envs/nlp/nltk_data/corpora> or in one of the other paths that nltk checks.

3) Download spacy data as:
  >> python -m spacy download en

--- Run the Training Procedure -------
0_data_preprocessing

1_optimal_topics

2_run_LDA 

(The last notebook does not need to be run, but if you skip this step then it is necessary to change the load statements throught the subsequent scripts.) 

--- Figures & Analysis ----------------





