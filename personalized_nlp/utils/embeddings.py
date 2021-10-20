from transformers import AutoTokenizer, AutoModel
import torch

from tqdm import tqdm
import pickle
import os


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

        mask = batch_encoding['input_ids'] > 0
        # all_embeddings.append(emb.pooler_output) ## podejscie nr 1 z tokenem CLS
        for i in range(emb[0].size()[0]):
            all_embeddings.append(emb[0][i, mask[i] > 0, :].mean(
                axis=0)[None, :])  # podejscie nr 2 z usrednianiem

    return torch.cat(all_embeddings, axis=0).to('cpu')


def create_embeddings(texts, embeddings_path=None,
                      model_name='xlm-roberta-base',
                      use_cuda=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if use_cuda:
        model = model.to('cuda')

    embeddings = _get_embeddings(texts, tokenizer, model, use_cuda=use_cuda)

    text_idx_to_emb = {}
    for i in range(embeddings.size(0)):
        text_idx_to_emb[i] = embeddings[i].numpy()

    if not os.path.exists(os.path.dirname(embeddings_path)):
        os.makedirs(os.path.dirname(embeddings_path))

    if embeddings_path:
        pickle.dump(text_idx_to_emb, open(embeddings_path, 'wb'))

    return text_idx_to_emb
