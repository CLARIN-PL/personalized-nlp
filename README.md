Project: nlperspectives

Method: USER_ID


## Important files in this branch:

- personalized_nlp/models/user_id.py	: Code for the 'userid' method (add the annotator_id as a special token on the embeddings without modifying the raw_texts or annotations)

- personalized_nlp/models/__init__.py	: Added mapping for the 'userid' method

- personalized_nlp/settings.py		: Added EMBEDDINGS_SIZES and TRANSFORMER_MODEL_STRINGS for 'roberta' (specifically, roberta-base)

- personalized_nlp/experiments/emopers_userid_notune.py	: A test-drive experiment without fine-tuning

- personalized_nlp/experiments/emopers_userid_tune.py		: The experiment involving fine-tuning (also here: https://colab.research.google.com/drive/1eZqlMPia3Vxn10kxosI3NgyMJ93uIyAX?usp=sharing)

- personalized_nlp/learning/train_alt.py : A customized version of train.py where a flag is added to choose running against test split or not, also added a method for getting the checkpoint directory

## To Do:

- a preprocessing step of averaging the embedding with attention mask > 0 before feeding into fc1

## Questions:
When I try to use some code from embeddings.py to average the embedding with attention mask > 0, (i.e. the line "mask = batch_encoding attention_mask > 0" and the loop "for i in range(emb....):....", I get the error saying "there are too many indices for tensor of dimension 1".

I tried changing the line "all_embeddings.append(emb\[0\]\[i, mask\[i\] > 0, :\].mean(axis=0)\[None, :\])" into "all_embeddings.append(emb\[0\]\[mask\[i\]].mean(axis=0)\[None, :\])" and I get another error saying that the mask's shape \[128\] does not match the tensor's shape \[768\].

How to do this correctly?
