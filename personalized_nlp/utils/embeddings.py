from typing import List
from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
import torch

from tqdm import tqdm
import pickle
import os
import fasttext
import numpy as np
from personalized_nlp.settings import CBOW_EMBEDDINGS_PATH, SKIPGRAM_EMBEDDINGS_PATH


def _get_embeddings(texts, tokenizer, model, max_seq_len=256, use_cuda=False):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    all_embeddings = []
    for batched_texts in tqdm(batch(texts, 200), total=len(texts)/200):
        with torch.no_grad():
            batch_encoding = tokenizer.batch_encode_plus(
                batched_texts,
                padding='longest',
                add_special_tokens=True,
                truncation=True, max_length=max_seq_len,
                return_tensors='pt',
            ).to(device)

            emb = model(**batch_encoding)

        mask = batch_encoding['attention_mask'] > 0

        for i in range(emb[0].size()[0]):
            all_embeddings.append(emb[0][i, mask[i] > 0, :].mean(
                axis=0)[None, :])  # podejscie nr 2 z usrednianiem

    return torch.cat(all_embeddings, axis=0).to('cpu')


def create_embeddings(texts, embeddings_path=None,
                      model_name='xlm-roberta-base',
                      use_cuda=True):

    if model_name == 'random':
        embeddings = torch.rand(len(texts), 768).numpy()
    elif model_name in ['skipgram', 'cbow']:
        embeddings = create_fasttext_embeddings(texts, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        if use_cuda:
            model = model.to('cuda')

        embeddings = _get_embeddings(texts, tokenizer, model, use_cuda=use_cuda)
        embeddings = embeddings.cpu().numpy()

    text_idx_to_emb = {}
    for i in range(embeddings.shape[0]):
        text_idx_to_emb[i] = embeddings[i]

    if not os.path.exists(os.path.dirname(embeddings_path)):
        os.makedirs(os.path.dirname(embeddings_path))

    if embeddings_path:
        pickle.dump(text_idx_to_emb, open(embeddings_path, 'wb'))

    return text_idx_to_emb


def create_fasttext_embeddings(texts: List[str], model_name: str):
    if model_name == 'skipgram':
        embeddings_path = SKIPGRAM_EMBEDDINGS_PATH
    else:
        embeddings_path = CBOW_EMBEDDINGS_PATH
        
    ft = fasttext.load_model(str(embeddings_path))
    
    embeddings = [ft.get_sentence_vector(t.replace('\n', ' ')) for t in texts]
    embeddings = np.array(embeddings)
    return embeddings
