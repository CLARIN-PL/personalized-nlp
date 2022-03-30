import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from personalized_nlp.settings import TRANSFORMER_MODEL_STRINGS

from typing import Dict, Any
import pytorch_lightning as pl

class NetUserID(pl.LightningModule):
    def __init__(self, output_dim, annotator_num, embedding_type, text_embedding_dim=768, model_name='roberta-base', max_length=128,
                 base_model=None, dp=0.35, dp_emb=0.2, embedding_dim=20, hidden_dim=100, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_type = embedding_type

        self.fc1 = nn.Linear(text_embedding_dim, output_dim) 
        
        additional_special_tokens = [f'_#{a_id}#_' for a_id in range(annotator_num)]
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        
        self._frozen = False

        if not self._frozen:
            if embedding_type in TRANSFORMER_MODEL_STRINGS:
                embedding_type = TRANSFORMER_MODEL_STRINGS[embedding_type]
            self.model = AutoModel.from_pretrained(embedding_type)
            self._tokenizer = AutoTokenizer.from_pretrained(embedding_type)
            self.max_length = max_length

        self._tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self._tokenizer))

        self.base_model = base_model
        
        self.dp = nn.Dropout(p=dp)
        self.dp_emb = nn.Dropout(p=dp_emb)

    def forward(self, features):
        if not self._frozen:
            batched_texts = features['raw_texts'].tolist()
            batch_encoding = self._tokenizer.batch_encode_plus(
                batched_texts,
                padding='longest',
                add_special_tokens=True,
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.model.device)

            emb = self.model(**batch_encoding).pooler_output
        else:
            emb = features['embeddings']

        if self.base_model is not None:
            features['embeddings'] = emb
            emb = self.base_model(features)

        x = emb
        
        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(torch.cat([x], dim=1))
        return x

    def freeze(self) -> None:
        for name, param in self.named_parameters():
            if 'classifier' not in name: 
                param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.named_parameters():
                if 'classifier' not in name: 
                    param.requires_grad = True
        self._frozen = False

    def on_epoch_start(self):
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()
    
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