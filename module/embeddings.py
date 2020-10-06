import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):


    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class EmbeddingPE(nn.Module):

    def __init__(self,embeddings:nn.Embedding,dropout,dim,pe=True):
        super(EmbeddingPE,self).__init__()
        self.embeddings = embeddings
        self.pe = pe
        if self.pe:
            self.positionencoding = PositionalEncoding(dropout,dim)

    def forward(self, input):
        if self.pe:
            return self.positionencoding(self.embeddings(input))
        else:
            return self.embeddings(input)
