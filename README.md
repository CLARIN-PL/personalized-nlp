Project: nlperspectives

## Download data

To download data, enter personalized_nlp folder and type in:

`dvc pull`

## How to run experiments

`python -m personalized_nlp.experiments.jester`

## How to add new dataset

Copy one of existings dataset classes (personalized_nlp/datasets/) and modify paths and settings. Next, copy one of the experiments (personalized_nlp/experiments/) and customize the settings.

## Important files for nlperspectives, method: USER_ID

- `personalized_nlp/models/user_id.py`	: Code for the 'userid' method (add the annotator_id as a special token on the embeddings without modifying the raw_texts or annotations)

- `personalized_nlp/experiments/emopers_userid_notune.py`	: The experiment on StudEmo with User-ID without fine-tuning

- `personalized_nlp/experiments/emopers_userid_tune.py`	: The experiment on StudEmo with User-ID with fine-tuning

- `personalized_nlp/experiments/goemo_userid_notune.py`	: The experiment on GoEmo with User-ID without fine-tuning

- `personalized_nlp/experiments/goemo_userid_tune.py.py`	: The experiment on GoEmo with User-ID with fine-tuning
