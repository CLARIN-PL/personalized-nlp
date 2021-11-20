from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from scipy.stats import entropy


def get_tokenized_texts(data):
    """ Tokenize comments"""
    data = data.copy()
    data['text_clean'] = data['text'].str.replace('NEWLINE_TOKEN', ' ')

    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(data.text_clean.tolist())

    text_tokenized = tokenizer.texts_to_sequences(data.text_clean.tolist())
    text_tokenized = pad_sequences(
        text_tokenized, maxlen=256, dtype='int32', padding='post', truncating='post', value=0.0)

    idx_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return tokenizer, text_tokenized, idx_to_word


def get_word_stats(texts, annotations, tokenized, idx_to_word, annotation_column):
    """ Calculate mean and std of scores per word in dataset """
    text_stats = annotations.groupby('text_id')[annotation_column].agg([
        'mean', 'std']).reset_index()

    #text_stats['entropy'] = annotations.groupby('text_id')[annotation_column].apply(lambda x : entropy(x.value_counts(), base=2)).values
    text_stats = texts.merge(text_stats)

    text_num = tokenized.shape[0]
    text_indices = np.arange(text_num)[:, None] * np.ones_like(tokenized)

    word_with_text_idx = np.vstack(
        [text_indices.flatten(), tokenized.flatten()]).astype(int)
    word_with_text_idx = word_with_text_idx[:, word_with_text_idx[1, :] != 0]

    word_with_text_df = pd.DataFrame(word_with_text_idx.T)
    word_with_text_df.columns = ['text_idx', 'word_id']
    word_with_text_df = word_with_text_df.drop_duplicates()

    result_df = word_with_text_df.merge(
        text_stats.reset_index(), left_on='text_idx', right_on='index')

    word_stats = result_df.groupby('word_id').agg('mean')
    word_stats['word_count'] = result_df.groupby(
        'word_id')['text_idx'].count().values
    word_stats = word_stats.loc[:, ['mean', 'std', 'word_count']]
    word_stats['word'] = [idx_to_word[x] for x in word_stats.index]

    return word_stats


def get_tokens_sorted(tokens, tokens_stats, num_tokens=10):
    """ Sort tokens by mean score and cut number of tokens """
    tokens_sorted = np.zeros((tokens.shape[0], num_tokens))
    mean_dict = tokens_stats['mean'].to_dict()

    for i in range(tokens.shape[0]):
        text_tokens = list(set(tokens[i]))
        text_tokens = sorted(
            text_tokens, key=lambda x: mean_dict.get(x, 0), reverse=True)

        tokens_sorted[i, :len(text_tokens[:num_tokens])
                      ] = text_tokens[:num_tokens]

    return tokens_sorted.astype(int)

def get_text_data(data, annotations, annotation_column,
                  min_word_count=100, min_std=0.27, words_per_text=10):
    tokenizer, text_tokenized, idx_to_word = get_tokenized_texts(data)
    
    word_stats = get_word_stats(
        data, annotations, text_tokenized, idx_to_word, annotation_column)

    selected_words_idx = word_stats[(word_stats.word_count > min_word_count) & (
        word_stats['std'] > min_std)].index.tolist()

    selected_words_tokenized = text_tokenized.copy()
    selected_words_tokenized[~np.isin(text_tokenized, selected_words_idx)] = 0

    tokens_sorted = get_tokens_sorted(
        selected_words_tokenized, word_stats, words_per_text)
    
    return tokenizer, text_tokenized, idx_to_word, tokens_sorted, word_stats.loc[selected_words_idx]
