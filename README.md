# humor-personalization
Personalized sense of humor prediction

## Download data

To download data, enter personalized_nlp folder and type in:

`dvc pull`

## How to run experiments:

`python -m personalized_nlp.experiments.jester`

## How to add new dataset

Copy one of existings dataset classes (personalized_nlp/datasets/) and modify paths and settings. Next, copy one of the experiments (personalized_nlp/experiments/) and customize the settings.