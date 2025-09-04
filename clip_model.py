import math

import torch
import torch.nn.functional as F
import torch.nn as nn

import resnet34embedder
import textembedder


class CLIP(nn.Module):
    def __init__(self, vocab_size, img_feat_dim, text_feat_dim, shared_dim):
        super().__init__()

        self.text_encoder = textembedder.TextRNN(vocab_size=vocab_size, text_feat_dim=text_feat_dim,dropout=0.1,pad_id=0)
        self.image_encoder = resnet34embedder.ResNet34Embedder(img_feat_dim)

        self.img_ln  = nn.LayerNorm(img_feat_dim)
        self.txt_ln  = nn.LayerNorm(text_feat_dim)
        self.image_projector = nn.Linear(img_feat_dim, shared_dim, bias=False)
        self.text_projector  = nn.Linear(text_feat_dim, shared_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
        # TODO: std 0.02 was recommended, but can we confirm that's good for the input dim size?
        nn.init.trunc_normal_(self.image_projector.weight, std=0.02)
        nn.init.trunc_normal_(self.text_projector.weight,  std=0.02)

    def encode(self, x_img, x_text, lengths):
        # returns L2-normalized embeddings
        # Note on image encoder: Input expected: (B, 3, 128, 128), that is channels is dimension 1. Output: B, img_feat_dim
        img = self.image_projector(self.img_ln(self.image_encoder(x_img)))
        # Note on text encoder:  Input expected: tokens (B,L), lengths (B). Output: B, text_feat_dim
        txt = self.text_projector(self.txt_ln(self.text_encoder(x_text, lengths)))
        img = F.normalize(img, p=2, dim=-1)
        txt = F.normalize(txt, p=2, dim=-1)
        return img, txt

    def forward(self, x_img, x_text, lengths):
        # keep forward as a convenience for non-queue eval
        img, txt = self.encode(x_img, x_text, lengths)
        scale = self.logit_scale.exp().clamp(max=100.0)
        # batch-only logits for quick eval
        logits_per_image = (img.float() @ txt.float().t()) * scale
        logits_per_text  = logits_per_image.t()
        return txt, img, logits_per_text, logits_per_image, scale