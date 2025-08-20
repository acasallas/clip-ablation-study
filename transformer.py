# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO:
#deal with EOT issue
# double check flash attention stuff, their params are always changing


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pad_token_id = pad_token_id

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Cache lower-triangular mask (True on allowed positions)
        causal = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.proj_out.is_res_init_scaling_needed = True  # used by custom init

        self.attn_dropout_p = dropout_rate

    def forward(self, x, input_ids):
        # x: (B, T, E), input_ids: (B, T)
        B, T, E = x.shape
        if T > self.causal_mask.size(0):
            raise ValueError(f"sequence length {T} exceeds cached context_len {self.causal_mask.size(0)}")

        qkv = self.qkv(x)                                   # (B, T, 3E)
        q, k, v = qkv.split(self.embed_dim, dim=-1)         # each (B, T, E)

        # (B, nh, T, dh)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Build boolean mask (True = allowed): causal AND non-PAD keys
        causal = self.causal_mask[:T, :T].to(x.device).unsqueeze(0).unsqueeze(0)   # (1,1,T,T)
        key_keep = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)      # (B,1,1,T)
        attn_mask = causal & key_keep                                              # (B,1,T,T), broadcast to (B,nh,T,T)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,                              # boolean: True = keep
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False
        )  # (B, nh, T, dh)

        y = y.transpose(1, 2).contiguous().view(B, T, E)      # (B, T, E)
        y = self.proj_out(y)
        y = self.dropout(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, context_len, pad_token_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate, context_len, pad_token_id)

        self.ln2 = nn.LayerNorm(embed_dim)
        ff_dim = 4 * embed_dim
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.ff2.is_res_init_scaling_needed = True
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, input_ids):
        x = x + self.attn(self.ln1(x), input_ids)
        x = x + self.dropout(self.ff2(F.gelu(self.ff1(self.ln2(x)))))
        return x


class Transformer(nn.Module):
    """
    CLIP text encoder:
      - Input:  input_ids (B, T), lengths (B)
      - Output: per-caption embedding (B, E)
      - Uses causal-style attention with EOT pooling (canonical CLIP behavior).
    """
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks, pad_token_id, eot_token_id):
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.pad_token_id = pad_token_id
        self.eot_token_id = eot_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self._prepare_classical_posemb(context_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout_rate, context_len, pad_token_id)
            for _ in range(n_blocks)
        ])

        # No LM head; we return a single vector (pooled)
        self.apply(self._init_weights)

    def _prepare_classical_posemb(self, context_len, embed_dim):
        pos = torch.arange(context_len, dtype=torch.float32)                            # (T,)
        inv = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        angles = pos[:, None] * inv[None, :]                                            # (T, E/2)
        pe = torch.zeros(context_len, embed_dim, dtype=torch.float32)                   # (T, E)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        self.register_buffer("pos_emb", pe.unsqueeze(0), persistent=False)              # (1, T, E)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if hasattr(m, "is_res_init_scaling_needed") and m.is_res_init_scaling_needed:
                std = std * ((2 * self.n_blocks) ** -0.5)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def _eot_index(self, input_ids, lengths=None):
        # Prefer explicit EOT; fall back to lengths-1 if missing.
        eot_mask = (input_ids == self.eot_token_id)            # (B, T)
        has_eot = eot_mask.any(dim=1)

        # TODO: what do we do if no EOT exists? I think we wanna return a default.
        idx = eot_mask.float().argmax(dim=1)
        return idx

    def forward(self, input_ids):
        # input_ids: (B, T), lengths: (B,)
        B, T = input_ids.shape
        if T > self.context_len:
            raise ValueError(f"seq len {T} exceeds context_len {self.context_len}")

        x = self.embedding(input_ids) * math.sqrt(self.embed_dim)         # (B, T, E)
        x = x + self.pos_emb[:, :T, :].to(x.dtype)
        x = self.emb_dropout(x)

        for blk in self.blocks:
            x = blk(x, input_ids)                                        # (B, T, E)

        eot_idx = self._eot_index(input_ids, lengths)                      # (B,)
        b = torch.arange(B, device=x.device)
        pooled = x[b, eot_idx, :]                                          # (B, E)
        return pooled
