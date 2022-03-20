Project: nlperspectives

Method: USER_ID


Important files in this branch:

- personalized_nlp/models/user_id.py	: Code for the 'userid' method (add the annotator_id as a special token on the embeddings without modifying the raw_texts or annotations)

- personalized_nlp/models/__init__.py	: Added mapping for the 'userid' method

- personalized_nlp/settings.py		: Added EMBEDDINGS_SIZES and TRANSFORMER_MODEL_STRINGS for 'roberta' (specifically, roberta-base)

- personalized_nlp/experiments/emopers_userid_notune.py	: A test-drive experiment without adding extra params to the train_test function

- personalized_nlp/experiments/emopers_userid_tune.py		: The experiment involving some extra params to the train_test function for fine-tuning


To Do:

- change the eval_loss function to be based on R2 (now apparently it is still based on MSELoss)

Questions:
1. Does models/user_id.py look fine? It seems to run without any issue, but I'm not sure if I configured everything correctly. The foundation of the code is from models/onehot.py but with some modification and addition (to add the annotator_id as a special token) from models/transformer.py.

2. The experiments keep crashing on Google Colab during the training (e.g. when epoch 0 is halfway or almost finished) due to an unknown cause. I'm not sure if this is due to GColab's limited resources or a bug in my code.

3. The function train_test() from learning/train.py seems to already take into account if we want to fine-tune the model. Does it mean that I just need to pass extra parameters to this function, i.e. the weight decay and the metric function, in order to fine-tune the model?

4. According to the explanation in the beginning (and from the output from the test run of my experiment), if we set regression = True then the architecture automatically uses R2 as the metric. But in the validation_step, the loss function is still based on MSELoss, so I think we must modify the validation_step's loss function? Bonus question: Apparently MSELoss is available as a built-in function, but R2-loss is not?



# humor-personalization
Personalized sense of humor prediction

## Download data

To download data, enter personalized_nlp folder and type in:

`dvc pull`

## How to run experiments:

`python -m personalized_nlp.experiments.jester`

## How to add new dataset

Copy one of existings dataset classes (personalized_nlp/datasets/) and modify paths and settings. Next, copy one of the experiments (personalized_nlp/experiments/) and customize the settings.