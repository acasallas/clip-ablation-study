import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.transforms as transforms
import wandb

import resnet34embedder


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class RingQueue:
    """Fixed-size ring buffer for L2-normalized embeddings. Used to implement XBM queues."""
    def __init__(self, dim: int, capacity: int, device: torch.device, dtype=torch.float16):
        self.dim = dim
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.buf = torch.zeros(capacity, dim, device=device, dtype=dtype)
        self.size = 0
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor):
        # x: (B, dim), assumed already L2-normalized
        b = x.shape[0]
        # if batch is larger than capacity, entire queue is filled up by most recent items in batch
        if b >= self.capacity:
            self.buf.copy_(x[-self.capacity:].to(self.dtype))
            self.size = self.capacity
            self.ptr = 0
            return
        # otherwise we fill up the queue in a ring fashion.
        end = self.ptr + b
        if end <= self.capacity:
            self.buf[self.ptr:end].copy_(x.to(self.dtype))
        else:
            first = self.capacity - self.ptr
            self.buf[self.ptr:].copy_(x[:first].to(self.dtype))
            self.buf[:end - self.capacity].copy_(x[first:].to(self.dtype))
        self.ptr = (self.ptr + b) % self.capacity
        self.size = min(self.size + b, self.capacity)

    def get(self) -> torch.Tensor:
        # Return the queue content in a contiguous (size, dim) tensor view
        # note the time order of the embeddings may be discontiguous, but for our purposes that is fine.
        return self.buf[:self.size]


class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        out_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Project to shared CLIP space (512 by default)
        proj_in = hidden_size * 2
        self.pre_ln = nn.LayerNorm(proj_in)
        self.proj = nn.Linear(proj_in, out_dim)

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
        returns: (B, out_dim) [optionally L2-normalized]
        """
        # lengths must be CPU ints for pack
        x = self.embedding(tokens)  # (B, L, E)

        packed = pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)   # h_n: (num_layers*num_directions, B, H)

        # take the last layer’s hidden state
        # concat forward and backward of last layer
        h_last_fwd = h_n[-2]  # (B, H)
        h_last_bwd = h_n[-1]  # (B, H)
        h = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # (B, 2H)

        h = self.pre_ln(h)
        out = self.proj(h)  # (B, out_dim)
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class CLIP(nn.Module):
    def __init__(self, img_dim, text_dim, shared_dim):
        super().__init__()

        self.text_encoder = RNN(text_dim)
        self.image_encoder = resnet34embedder.ResNet34Embedder(img_dim)

        self.img_ln  = nn.LayerNorm(img_dim)
        self.txt_ln  = nn.LayerNorm(text_dim)
        self.image_projector = nn.Linear(img_dim, shared_dim, bias=False)
        self.text_projector  = nn.Linear(text_dim, shared_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
        # TODO: std 0.02 was recommended, but can we confirm that's good for the input dim size?
        nn.init.trunc_normal_(self.image_projector.weight, std=0.02)
        nn.init.trunc_normal_(self.text_projector.weight,  std=0.02)

    def encode(self, x_img, x_text):
        # returns L2-normalized embeddings
        img = self.image_projector(self.img_ln(self.image_encoder(x_img)))
        txt = self.text_projector(self.txt_ln(self.text_encoder(x_text)))
        img = F.normalize(img, p=2, dim=-1)
        txt = F.normalize(txt, p=2, dim=-1)
        return img, txt

    def forward(self, x_img, x_text):
        # keep forward as a convenience for non-queue eval
        img, txt = self.encode(x_img, x_text)
        scale = self.logit_scale.exp().clamp(max=100.0)
        # batch-only logits for quick eval
        logits_per_image = (img.float() @ txt.float().t()) * scale
        logits_per_text  = logits_per_image.t()
        return txt, img, logits_per_text, logits_per_image, scale

def clip_loss(logits_per_text, logits_per_image):
    B = logits_per_text.size(0)
    labels = torch.arange(B, device=logits_per_text.device)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    loss_i = F.cross_entropy(logits_per_image, labels)
    return 0.5 * (loss_t + loss_i)


def compute_logits_with_queue(img_emb, txt_emb, q_img, q_txt, scale, chunk=8192):
    """
    img_emb, txt_emb: (B, D) L2-normalized
    q_img, q_txt: (N, D) L2-normalized (may be empty)
    scale: scalar tensor
    Returns logits_per_text, logits_per_image using [batch + queue] negatives.
    """
    B = img_emb.size(0)
    device = img_emb.device

    # Concatenate current batch + queue
    all_txt = txt_emb if q_txt.size(0) == 0 else torch.cat([txt_emb, q_txt.to(dtype=txt_emb.dtype)], dim=0)
    all_img = img_emb if q_img.size(0) == 0 else torch.cat([img_emb, q_img.to(dtype=img_emb.dtype)], dim=0)

    # Compute in fp32 for numerical stability # TODO: think about whether this is needed.
    img32 = img_emb.float()
    txt32 = txt_emb.float()
    all_txt32 = all_txt.float()
    all_img32 = all_img.float()

    # we perform a chunked matrix multiplication. Why? Because it uses less VRAM.
    def matmul_chunked(A, Bt, chunk):
        # A: (B, D), Bt: (D, N) -> (B, N)
        N = Bt.size(1)
        out = torch.empty(A.size(0), N, device=A.device, dtype=torch.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            out[:, s:e] = A @ Bt[:, s:e]
        return out

    logits_per_image = matmul_chunked(img32, all_txt32.t(), chunk) * scale
    logits_per_text  = matmul_chunked(txt32, all_img32.t(), chunk) * scale

    return logits_per_text, logits_per_image


def train_clip():
    model = CLIP() # TODO: put parameters in.
    model.to(device)

    shared_dim = 512  # must match your projector dim
    K = 16384
    queue_chunk_size = 8192
    # Set up XBM queues.
    q_img = RingQueue(dim=shared_dim, capacity=K, device=device, dtype=torch.float16)
    q_txt = RingQueue(dim=shared_dim, capacity=K, device=device, dtype=torch.float16)
    ce = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss, count = 0.0, 0
        for batch in training_loader:
            optimizer.zero_grad(set_to_none=True)

            images, token_ids, lengths = batch
            lengths = lengths.to(device, non_blocking=True)

            images    = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            token_ids = token_ids.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_emb, txt_emb = model.encode(images, token_ids)  # L2-normalized (B, D)
                # temperature param stays fp32 in the module; read the scale here
                scale = model.logit_scale.exp().clamp(max=100.0)

            logits_t, logits_i = compute_logits_with_queue(
                img_emb, txt_emb, q_img.get(), q_txt.get(), scale, chunk=queue_chunk_size
            )

            B = images.size(0)
            target = torch.arange(B, device=device)
            loss = 0.5 * (ce(logits_i, target) + ce(logits_t, target))

            loss.backward()
            optimizer.step()

            # Update queues (store as fp16 to save VRAM; inputs are already unit-norm)
            q_img.enqueue(img_emb.detach())
            q_txt.enqueue(txt_emb.detach())

            running_loss += loss.item() * B
            count += B

        print(f"epoch {epoch}: train_loss={running_loss / max(1, count):.4f}")


        # remember model.train() and model.eval()!
        model.eval()
        val_losses = 0
        val_correct = 0
        val_total = 0

        # TODO: update the eval for CLIP! Might have to be every few samples because might take too long to get around one epoch!
        with torch.no_grad():
            for i,data in enumerate(validation_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) # this doesn't work because outputs is a 10 vector!
                _, predicted = torch.max(outputs, 1)
                loss =loss_fn(outputs,labels)
                val_losses += loss.item()*inputs.size(0)
                val_correct += (labels==predicted).sum().item()
                val_total += labels.size(0)

        print(f"validation loss {val_losses/val_total} accuracy {val_correct/val_total}")
        metrics.update({"val_loss": train_losses/train_total})


if __name__ == "__main__":
    train_clip()