# toxicity
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotated_comments.tsv' -ac='worker_id' -tc='rev_id'

# attack
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/attack_annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/attack_annotated_comments.tsv' -ac='worker_id' -tc='rev_id'


# aggression
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/aggression_annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/aggression_annotated_comments.tsv' -ac='worker_id' -tc='rev_id'


# unhealthy conversations
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/unhealthy_conversations/uc_annotations.csv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/unhealthy_conversations/uc_texts.csv' -ac='annotator_id' -tc='text_id'

# measuring hate speech
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/measuring_hate_speech/annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/measuring_hate_speech/data.tsv' -ac='annotator_id' -tc='text_id'

# jester
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/jester/jester_annotations.csv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/jester/data.csv' -ac='annotator_id' -tc='text_id'

# humor
python3 -m scripts.process_dataframe -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/humor/texts/annotations.csv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/humor/texts/data.csv' -ac='annotator_id' -tc='text_id'

# humicroedit
python3 -m scripts.process_dataframe  -ac='user_id' -tc='text_id' -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/humicroedit/annotations.csv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/humicroedit/data.csv'

# clarin emo text
python3 -m scripts.process_dataframe  -ac='user_id' -tc='text_id' -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/clarin_emo_text/annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/clarin_emo_text/data.tsv'

# clarin emo sent
python3 -m scripts.process_dataframe  -ac='user_id' -tc='text_id' -ap='/home/konradkaranowski/storage/personalized-nlp/storage/data/clarin_emo_sent/annotations.tsv' -tp='/home/konradkaranowski/storage/personalized-nlp/storage/data/clarin_emo_sent/data.tsv'
