import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnotatorBiasNet(nn.Module):
    def __init__(self, output_dim, text_embedding_dim, word_num, annotator_num, 
                    dp=0.0, hidden_dim=100, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
      
        self.annotator_biases = torch.nn.Embedding(num_embeddings=annotator_num, 
                                                       embedding_dim=output_dim, 
                                                       padding_idx=0)
        self.word_biases = torch.nn.Embedding(num_embeddings=word_num, 
                                                       embedding_dim=output_dim, 
                                                       padding_idx=0)
      
        self.annotator_biases.weight.data.uniform_(-.001, .001)
        self.word_biases.weight.data.uniform_(-.001, .001)

        self.dp = nn.Dropout(p=dp)
        
        self.fc1 = nn.Linear(self.text_embedding_dim, self.hidden_dim) 
        self.fc2 = nn.Linear(self.hidden_dim, output_dim) 

        self.softplus = nn.Softplus()

    def forward(self, features):
        x = features['embeddings']
        annotator_ids = features['annotator_ids']
        tokens = features['tokens_sorted']

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)
        x = self.softplus(x)

        annotator_bias = self.annotator_biases(annotator_ids+1)
        word_biases = self.word_biases(tokens)

        mask = tokens != 0
        word_biases = (word_biases*mask[:, :, None]).sum(dim=1)

        x = self.fc2(x) + annotator_bias + word_biases
        
        return x