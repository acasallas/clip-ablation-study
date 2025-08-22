import datetime
from functools import lru_cache, partial
import math
import os
import random

from datasets import load_dataset
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import wandb

import resnet34embedder
import textembedder



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Will use device {device}.")


# TODO: torch.compile --> that is REALLY important, it could double your training efficiency.
# TODO: alan, you need to search to see what batch size is best for you.
# TODO: should you train a batch to see loss go down?



# TODO: profile and seriously consider pre-tokenizing dataset.
# think about how you can time the file throughout.
# TODO: may want to ask about sanity checks (like what you were being recommended to do with tokenization).
# TODO: strongly consider doing a hyperparameter search for like 1000 steps or something.
# TODO: look into some easy low-risk things to avoid bottlenecks are (like nonblocking=True, etc)


IMG_SIZE = 128 # clipping to 128 to better fit in a 4090 GPU
# We use the numbers from the CLIP dataset, should be close enough for BN to do its work.
# May want to review this if we struggle during training.
MEAN =[0.48145466, 0.4578275, 0.40821073]
STD =[0.26862954, 0.26130258, 0.27577711]


def to_pil_rgb(x):
    if isinstance(x, Image.Image):
        im = x
    else:
        im = Image.fromarray(x)
    return im if im.mode == "RGB" else im.convert("RGB")

train_transform = T.Compose([
    T.Lambda(to_pil_rgb),
    # TODO: you've been told you should resize and randomcrop
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),     # shortest side -> 128
    T.RandomCrop(IMG_SIZE),                                          # 128x128 crop
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05), # little color jitter, why not
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_transform = T.Compose([
    T.Lambda(to_pil_rgb),
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])



TOKENIZER_JSON_PATH = "clip_bpe/tokenizer.json"

@lru_cache(maxsize=None)
def _load_tok(path: str, max_len: int):
    tok = Tokenizer.from_file(path)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    fast.model_max_length = max_len
    fast.padding_side = "right"
    return fast

def collate_fn_top(examples, tokenizer_json_path: str, transform, max_len: int):
    # images
    imgs = [transform(ex["jpg"]) for ex in examples]
    images = torch.stack(imgs, dim=0)

    # text
    caps = [ex["txt"] if ex["txt"] is not None else "" for ex in examples]
    tok = _load_tok(tokenizer_json_path, max_len)
    enc = tok(caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].long()
    attention = enc["attention_mask"].long()
    lengths = attention.sum(dim=1).clamp_(min=1)  # guard against 0

    return images, input_ids, lengths


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


def compute_logits_with_queue(img_emb, txt_emb, q_img, q_txt, scale):
    all_txt = txt_emb if q_txt.size(0) == 0 else torch.cat([txt_emb, q_txt.to(dtype=txt_emb.dtype)], dim=0)
    all_img = img_emb if q_img.size(0) == 0 else torch.cat([img_emb, q_img.to(dtype=img_emb.dtype)], dim=0)

    img32, txt32 = img_emb.float(), txt_emb.float()
    all_txt32, all_img32 = all_txt.float(), all_img.float() # TODO: do we need to use tf32 here if we already set high precision?

    logits_per_image = img32 @ all_txt32.t()
    logits_per_text  = txt32  @ all_img32.t()
    return logits_per_text * scale, logits_per_image * scale


def param_groups(model, wd=0.1):
    """
    This function applied weight decay to all appropriate layers (and skips those that are not).
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if p.ndim == 1 or n.endswith(".bias") or "logit_scale" in n:
            no_decay.append(p)   # norms, biases, temperature
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, extra: dict = None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # restore random number generator - seems like a good idea.
    if "torch_rng_state" in ckpt: torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    if "numpy_rng_state" in ckpt: np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt: random.setstate(ckpt["python_rng_state"])

    start_epoch  = int(ckpt.get("epoch", 0))
    global_step  = int(ckpt.get("global_step", 0))
    extra        = ckpt.get("extra", {})
    return start_epoch, global_step, extra


IMG_COL = "jpg"
TXT_COL = "txt"
MAX_LEN=160


def print_clip_model_summary(model, batch_size, vocab_size, L=160):
    model.eval()
    dev = next(model.parameters()).device
    x_img = torch.randn(batch_size, 3, 128, 128, device=dev)
    x_txt = torch.randint(0, vocab_size, (batch_size, L), device=dev)
    lengths = torch.randint(1, L+1, (batch_size,), device=dev)
    summary(model, input_data=(x_img, x_txt, lengths),
            dtypes=[torch.float32, torch.long, torch.long])



def train_clip(resume_path=None):
    """
    resume_path - set only if we need to continue from a checkpoint.
    """
    torch.set_float32_matmul_precision('high')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config = {
        "img_feat_dim": 512,
        "text_feat_dim": 512,
        "shared_embedding_dim": 512,
        "xbm_size": 8192,
        "clip_batch_size": 2048,
        "learning_rate": 1e-4, # recommended for CLIP was 5e-4 * (batch_size / 512)
        "weight_decay": 0.1
    }

    use_xbm_queue = False

    save_dir="./third_try_ckpts"
    # um... with a batch of 512 3M will only have 5859 steps. We still have to time it though.
    save_every_steps = 500
    eval_every_steps = 500
    print_every_steps = 100
    dataloader_num_workers = 4 # tune this to the PC CPU.
    num_epochs = 30

    with wandb.init(config=config,project="clip-custom-experiment",entity="alancasallas-self") as run:
        dataset = load_dataset("pixparse/cc3m-wds")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

        train_collate = partial(
            collate_fn_top, tokenizer_json_path=TOKENIZER_JSON_PATH,
            transform=train_transform, max_len=MAX_LEN,
        )
        val_collate = partial(
            collate_fn_top, tokenizer_json_path=TOKENIZER_JSON_PATH,
            transform=val_transform, max_len=MAX_LEN,
        )

        training_loader = DataLoader(
            train_dataset, batch_size=wandb.config.clip_batch_size, shuffle=True,
            drop_last=True, num_workers=dataloader_num_workers, pin_memory=True,
            persistent_workers=dataloader_num_workers > 0, collate_fn=train_collate,
        )

        validation_loader = DataLoader(
            val_dataset, batch_size=wandb.config.clip_batch_size, shuffle=False,
            drop_last=False, num_workers=dataloader_num_workers, pin_memory=True,
            persistent_workers=dataloader_num_workers > 0, collate_fn=val_collate,
        )

        tok = _load_tok(TOKENIZER_JSON_PATH, MAX_LEN)

        model = CLIP(tok.vocab_size, wandb.config.img_feat_dim, wandb.config.text_feat_dim, wandb.config.shared_embedding_dim)
        model.to(device)
        print_clip_model_summary(model, min(8, wandb.config.clip_batch_size), tok.vocab_size)

        # Warmup + cosine (to ~1e-6). ramp up over 2k steps:
        warmup_steps = 1000
        total_steps = len(training_loader) * num_epochs
        base_lr = wandb.config.learning_rate

        # the famous cosine annealing with rampup.
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # cosine from 1.0 -> ~1e-6/base_lr
            min_lr = 1e-6 / base_lr
            return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * t))

        optimizer = AdamW(param_groups(model, wd=wandb.config.weight_decay), lr=wandb.config.learning_rate, betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Load saved checkpoint if requested
        os.makedirs(save_dir, exist_ok=True)
        start_epoch, global_step, best_val = 0, 0, float("inf")
        if resume_path and os.path.isfile(resume_path):
            start_epoch, global_step, extra = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=str(device))
            best_val = extra.get("best_val", best_val)
            print(f"Resumed from {resume_path} at epoch={start_epoch}, global_step={global_step}")

        # Set up XBM queues.
        q_img = RingQueue(dim=wandb.config.shared_embedding_dim, capacity=wandb.config.xbm_size, device=device, dtype=torch.float16)
        q_txt = RingQueue(dim=wandb.config.shared_embedding_dim, capacity=wandb.config.xbm_size, device=device, dtype=torch.float16)
        ce = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, num_epochs):
            running_train_loss, running_train_img_correct, running_train_txt_correct, running_train_count = 0.0, 0, 0, 0
            for step_number, batch in enumerate(training_loader):
                model.train()
                optimizer.zero_grad(set_to_none=True)

                images, token_ids, lengths = batch
                lengths = lengths.to(device, non_blocking=True)
                images    = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                token_ids = token_ids.to(device, non_blocking=True)

                if global_step == 1:
                    print("lengths stats:", lengths.min().item(), lengths.float().mean().item(), lengths.float().std().item(), lengths.max().item())

                use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
                autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16 if device.type == "cuda" else torch.float32

                if use_xbm_queue:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        img_emb, txt_emb = model.encode(images, token_ids, lengths)
                        scale = model.logit_scale.exp().clamp(max=100.0)
                    logits_t, logits_i = compute_logits_with_queue(img_emb, txt_emb, q_img.get(), q_txt.get(), scale)
                else:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        _, _, logits_t, logits_i, scale = model(images, token_ids, lengths)  # in-batch only

                B = images.size(0)
                target = torch.arange(B, device=device)

                # CLIP loss
                loss = 0.5 * (ce(logits_i, target) + ce(logits_t, target))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update queues (store as fp16 to save VRAM; inputs are already unit-norm)
                if use_xbm_queue:
                    q_img.enqueue(img_emb.detach())
                    q_txt.enqueue(txt_emb.detach())


                # TODO: to keep in mind: this accuracy will be lower because we're including the negatives from the queue. 
                # metrics
                with torch.no_grad():
                    img_preds = logits_i.argmax(dim=-1)
                    txt_preds = logits_t.argmax(dim=-1)
                    running_train_loss        += loss.item() * B
                    running_train_img_correct += (img_preds == target).sum().item()
                    running_train_txt_correct += (txt_preds == target).sum().item()
                    running_train_count       += B

                total_train_loss = running_train_loss/running_train_count
                total_train_img_accuracy = running_train_img_correct/running_train_count
                total_train_txt_accuracy = running_train_txt_correct/running_train_count

                global_step += 1

                # TODO: consider logging scale.

                # print every few steps to show we are making progress
                if global_step % print_every_steps == 0:
                    print(f"[{datetime.datetime.now()}] Global step: {global_step} Training Loss: {total_train_loss:.4f} img logit accuracy {total_train_img_accuracy:.4f} text logit accuracy {total_train_txt_accuracy:.4f} scale {scale.item():.2f}")

                # save every few steps
                if global_step % save_every_steps == 0:
                    ckpt_path = os.path.join(save_dir, f"step_{global_step}.pth")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler, epoch, global_step,
                        extra={"best_val": best_val}
                    )
                    print(f"Checkpoint saved {ckpt_path}")

                # TODO: we'll have to time how long it takes to eval entire validation set.
                if global_step % eval_every_steps == 0:
                    metrics = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_num": step_number,
                        "train_loss": total_train_loss,
                        "train_img_accuracy": total_train_img_accuracy,
                        "train_txt_accuracy": total_train_txt_accuracy
                    }
                    model.eval()
                    with torch.no_grad():
                        running_val_loss = running_val_img_correct = running_val_txt_correct = running_val_count = 0
                        for _, (images, token_ids, lengths) in enumerate(validation_loader):
                            lengths  = lengths.to(device, non_blocking=True)
                            images   = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                            token_ids= token_ids.to(device, non_blocking=True)
                            B = images.size(0)

                            # TODO: do we have to use bf16 here?
                            txt_emb, img_emb, logits_t, logits_i, _ = model(images, token_ids, lengths)
                            target = torch.arange(B, device=device)
                            loss = 0.5 * (ce(logits_i, target) + ce(logits_t, target))

                            img_preds = logits_i.argmax(dim=-1)
                            txt_preds = logits_t.argmax(dim=-1)
                            running_val_loss        += loss.item() * B
                            running_val_img_correct += (img_preds == target).sum().item()
                            running_val_txt_correct += (txt_preds == target).sum().item()
                            running_val_count       += B

                    total_val_loss          = running_val_loss / running_val_count
                    total_val_img_accuracy  = running_val_img_correct / running_val_count
                    total_val_txt_accuracy  = running_val_txt_correct / running_val_count
                    metrics.update({
                        "val_loss": total_val_loss,
                        "val_img_accuracy": total_val_img_accuracy,
                        "val_txt_accuracy": total_val_txt_accuracy
                    })
                    wandb.log(metrics)

                    # TODO: not sure if we'll need this, we may just save every time we eval and choose best one.
                    #if current_val_loss < best_val:
                    #    best_val = current_val_loss:
                    #    best_path = os.path.join(save_dir, "best.pth")
                    #    save_checkpoint(best_path, model, optimizer, scheduler, epoch+1, global_step, extra={"best_val": best_val})
                    #    print(f"[ckpt] saved {best_path} (rolling best/last)")

        # final - probably won't get to this point.
        final_path = os.path.join(save_dir, "final.pth")
        save_checkpoint(final_path, model, optimizer, scheduler, num_epochs, global_step, extra={"best_val": best_val})
        print(f"Final checkpoint saved {final_path}")


if __name__ == "__main__":
    train_clip()