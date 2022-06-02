## Download data

To download data, enter personalized_nlp folder and type in:

`dvc pull`

## How to run experiments

`python -m personalized_nlp.experiments.jester`

## How to add new dataset

Copy one of existings dataset classes (personalized_nlp/datasets/) and modify paths and settings. Next, copy one of the experiments (personalized_nlp/experiments/) and customize the settings.

## NLPerspectives Workshop

The branch contains methods for paper _StudEmo: A Non-aggregated Review Dataset for Personalized Emotion Recognition_ 
presented during [NLPerspectives](https://nlperspectives.di.unito.it/) -- 1st Workshop 
on Perspectivist Approaches to NLP during [LREC 2022](https://lrec2022.lrec-conf.org/en/).

The used methods:
- AVG-ANN  -- using the baseline model implemented in `personalized_nlp/models/baseline.py`
- SINGLE-ANN  -- using the baseline model implemented in `personalized_nlp/models/user_id.py`
- USER_ID -- implemented in `personalized_nlp/models/user_id.py`
- Past Embeddings  -- implemented in `personalized_nlp/models/past_embeddings.py`
- HuBi Medium  -- implemented in `personalized_nlp/models/hubi_med_finetune.py`

The experiments are located in `personalized_nlp/experiments`, e.g,:
- `personalized_nlp/experiments/emopers_userid_notune.py` -- the experiment on StudEmo with User-ID without fine-tuning
- `personalized_nlp/experiments/emopers_userid_tune.py`	-- the experiment on StudEmo with User-ID with fine-tuning
- `personalized_nlp/experiments/goemo_userid_notune.py`	-- the experiment on GoEmo with User-ID without fine-tuning
- `personalized_nlp/experiments/goemo_userid_tune.py.py` -- the experiment on GoEmo with User-ID with fine-tuning
