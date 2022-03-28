import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from personalized_nlp.settings import TRANSFORMER_MODEL_STRINGS
from personalized_nlp.utils import tokenizer

class HuBiMedium(nn.Module):
    def __init__(self, output_dim, text_embedding_dim, word_num, annotator_num, embedding_type,
                 max_seq_len=128, dp=0.35, dp_emb=0.2, embedding_dim=20, hidden_dim=100, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_type = embedding_type

        self._frozen = False

        if not self._frozen:
            if embedding_type in TRANSFORMER_MODEL_STRINGS:
                embedding_type = TRANSFORMER_MODEL_STRINGS[embedding_type]
            self._model = AutoModel.from_pretrained(embedding_type)
            self._tokenizer = AutoTokenizer.from_pretrained(embedding_type)
            self.max_seq_len = max_seq_len
      
        self.word_biases = torch.nn.Embedding(num_embeddings=word_num, 
                                                  embedding_dim=output_dim, 
                                                  padding_idx=0)
              
        self.annotator_embeddings = torch.nn.Embedding(num_embeddings=annotator_num, 
                                                       embedding_dim=self.embedding_dim, 
                                                       padding_idx=0)

        self.annotator_embeddings.weight.data.uniform_(-.01, .01)
        self.word_biases.weight.data.uniform_(-.01, .01)

        self.dp = nn.Dropout(p=dp)
        self.dp_emb = nn.Dropout(p=dp_emb)
        
        self.fc1 = nn.Linear(self.text_embedding_dim, self.hidden_dim) 
        self.fc2 = nn.Linear(self.hidden_dim, output_dim) 
        self.fc_annotator = nn.Linear(self.embedding_dim, self.hidden_dim) 

        self.softplus = nn.Softplus()

    def forward(self, features):
        if not self._frozen:
            batched_texts = features['raw_texts'].tolist()
            batch_encoding = self._tokenizer.batch_encode_plus(
                batched_texts,
                padding='longest',
                add_special_tokens=True,
                truncation=True, 
                max_length=self.max_seq_len,
                return_tensors='pt',
            ).to(self._model.device)

            x = self._model(**batch_encoding).pooler_output
        else:
            x = features['embeddings']

        annotator_ids = features['annotator_ids'].long()
        tokens = features['tokens_sorted'].long()

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)
        x = self.softplus(x)

        word_biases = self.word_biases(tokens)

        mask = tokens != 0
        word_biases = (word_biases*mask[:, :, None]).sum(dim=1)
        
        annotator_embedding = self.annotator_embeddings(annotator_ids+1)
        annotator_embedding = self.dp_emb(annotator_embedding)
        annotator_embedding = self.softplus(self.fc_annotator(annotator_embedding))

        x = self.fc2(x * annotator_embedding) + word_biases
        
        return x

    def freeze(self) -> None:
        for name, param in self.named_parameters():
            if 'classifier' not in name: 
                param.requires_grad = False
                print('Freezing parameter:', name)
        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.named_parameters():
                if 'classifier' not in name: 
                    param.requires_grad = True
                    print('Unfreezing parameter:', name)
        self._frozen = False

    def on_epoch_start(self):
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()
