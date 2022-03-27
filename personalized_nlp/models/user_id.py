import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from typing import Dict, Any
import pytorch_lightning as pl

class NetUserID(pl.LightningModule):
    def __init__(self, output_dim, annotator_num, text_embedding_dim=768, model_name='roberta-base', max_length=128,
                 base_model=None, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim, output_dim) 
        
        additional_special_tokens = [f'_#{a_id}#_' for a_id in range(annotator_num)]
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.add_special_tokens(special_tokens_dict)

        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self._tokenizer))

        self.max_length = max_length
        self.base_model = base_model

    def forward(self, features):
        texts_raw = features['raw_texts'].tolist()

        tokenizer = self._tokenizer
        model = self.model

        batch_encoding = tokenizer.batch_encode_plus(
            texts_raw,
            padding='longest',
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        batch_encoding = batch_encoding.to('cuda')

        emb = model(**batch_encoding).pooler_output

        if self.base_model is not None:
            features['embeddings'] = emb
            emb = self.base_model(features)

        x = emb
        
        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(torch.cat([x], dim=1))
        return x

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir, output_dim, text_embedding_dim,
    word_num, annotator_num, dp, dp_emb, embedding_dim, hidden_dim, bias_vector_length, map_location=None):
        """
        Primary way of loading model from a checkpoint
        :param checkpoint_path:
        :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
        :return:
        """
        checkpoint_path = checkpoint_dir / 'bestmodel.ckpt'
        
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # try:
        #     ckpt_hparams = checkpoint['hparams']
        # except KeyError:
        #     raise IOError(
        #         "Checkpoint does not contain hyperparameters. Are your model hyperparameters stored"
        #         "in self.hparams?"
        #     )
        # hparams = Namespace(**ckpt_hparams)

        # load the state_dict on the model automatically
        model = cls(output_dim=output_dim,
                text_embedding_dim=text_embedding_dim,
                word_num=word_num,
                annotator_num=annotator_num,
                dp=dp,
                dp_emb=dp_emb,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                bias_vector_length=bias_vector_length)
        model.load_state_dict(torch.load(checkpoint_dir / 'state_dict'))

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)
        
        model.eval()

        return model