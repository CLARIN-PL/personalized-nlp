# personalized-nlp
Personalized prediction applied to various subjective natural language processing (NLP) tasks

## Download data

To download data, enter personalized_nlp folder and type in:

`dvc pull`

## How to run experiments:

First, preprocess selected dataset with `scripts.process_data` pipeline, to assigns folds to texts:

`python -m scripts.process_data --annotations_df_path .../personalized-nlp/storage/data/unhealthy_conversations/uc_annotations.csv --texts_df_path .../personalized-nlp/storage/data/unhealthy_conversations/uc_texts.csv --annotator_col annotator_id --num_folds 5 --text_col text_id`

Then, you can run experiments with:

`python -m personalized_nlp.experiments.unhealthy`
`python -m personalized_active_learning.experiments.unhealthy`

## How to add DataModule for a new dataset

Copy one of the existing dataset classes (personalized_nlp/datasets/) and modify paths and settings. Next, copy one of the experiments (personalized_nlp/experiments/) and customize the settings.

## How to select folding setup

Set the `stratify_folds_by` argument in datamodule: `None` for standard train-val-test split, `'users'` for users/past-present-future1-future2 split and `'texts'` for texts folds.
