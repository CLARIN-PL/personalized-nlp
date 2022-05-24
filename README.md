Project: nlperspectives

Method: USER_ID


## Important files in this branch:

- `personalized_nlp/models/user_id.py`	: Code for the 'userid' method (add the annotator_id as a special token on the embeddings without modifying the raw_texts or annotations)

- `personalized_nlp/models/__init__.py`	: Added mapping for the 'userid' method

- `personalized_nlp/settings.py`	: Added EMBEDDINGS_SIZES and TRANSFORMER_MODEL_STRINGS for 'roberta' (specifically, roberta-base), added GO_EMOTIONS_LABELS for the GoEmo experiments

- `personalized_nlp/experiments/emopers_userid_notune.py`	: The experiment on StudEmo with User-ID without fine-tuning

- `personalized_nlp/experiments/emopers_userid_tune.py`	: The experiment on StudEmo with User-ID with fine-tuning

- `personalized_nlp/experiments/goemo_userid_notune.py`	: The experiment on GoEmo with User-ID without fine-tuning

- `personalized_nlp/experiments/goemo_userid_tune.py.py`	: The experiment on GoEmo with User-ID with fine-tuning
