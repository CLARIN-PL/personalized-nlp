import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnotatorEmbeddingNet(nn.Module):

    def __init__(self, output_dim, text_embedding_dim, word_num, annotator_num,
                 dp=0.35, dp_emb=0.2, embedding_dim=20, hidden_dim=100, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
      
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
        x = features['embeddings']
        annotator_ids = features['annotator_ids']
        tokens = features['tokens_sorted']

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
