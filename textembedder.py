import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextRNN(nn.Module):
    def __init__(
        self, vocab_size: int, text_feat_dim: int = 512,
        dropout: float = 0.1, pad_id: int = 0,
    ):
        super().__init__()
        embed_dim = 256
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=text_feat_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Project to shared CLIP space (512 by default)
        proj_in = text_feat_dim * 2
        self.pre_ln = nn.LayerNorm(proj_in)
        self.proj = nn.Linear(proj_in, text_feat_dim)

        self._init_weights()

    def _init_weights(self):
        # Embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # GRU params: Kaiming for input-hidden, orthogonal for hidden-hidden
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Projection
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """
        tokens:  (B, L) int64
        lengths: (B,)    int64, unpadded lengths
        returns: (B, text_feat_dim)
        """
        x = self.embedding(tokens)  # (B, L, E)

        packed = pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)   # h_n: (num_layers*num_directions, B, H)

        # take the last layer’s hidden state
        # concat forward and backward of last layer
        h_last_fwd = h_n[-2]  # (B, H)
        h_last_bwd = h_n[-1]  # (B, H)
        h = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # (B, 2H)

        h = self.pre_ln(h)
        out = self.proj(h)  # (B, text_feat_dim)
        return out