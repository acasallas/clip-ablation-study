import argparse
import io
import math
import os
import random
import urllib.request
from functools import lru_cache

from datasets import load_dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from clip_model import CLIP


classes = []
with open("imagenet_descriptions.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i%2 == 0:
            classes.append(lines[i].strip().lower())
        else:
            if line[0].lower() == "a":
                classes[-1] = classes[-1] + " is " + line.strip().lower()
            else:
                classes[-1] = classes[-1] + " " + line.strip().lower()
imagenet_class_prompts = classes
print(imagenet_class_prompts[200:205])



# ---------------- Device ----------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Will use device {device}.")


# ---------------- Constants & Config ----------------
IMG_SIZE = 128
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
TOKENIZER_JSON_PATH = "clip_bpe/tokenizer.json"
MAX_LEN = 160  # adjust to your text encoder limit
NUM_IMAGENET_CLASSES = 1000

TEMPLATES = [
    "an image of a {}",
    "an image of the {} .",
    "a {}",
    "a photo of a {} .",
    "a photo of the {}",
    "a picture of a {} .",
    "a picture of the {}",
]


# ---------------- Helpers ----------------
def to_pil_rgb(x):
    if isinstance(x, Image.Image):
        im = x
    else:
        im = Image.fromarray(x)
    return im if im.mode == "RGB" else im.convert("RGB")


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


def build_random_template_prompts(imagenet_classes):
    # For each class id c in [0..999], pick a random template and fill in the class name.
    return imagenet_class_prompts
    #prompts = [TEMPLATES[random.randrange(len(TEMPLATES))].format(imagenet_classes[c])
    #           for c in range(NUM_IMAGENET_CLASSES)]
    #return prompts


# ---------------- Main ----------------
def main(run_name: str, checkpoint_name: str):
    # Load ImageNet class names (index order must match labels)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    imagenet_classes = [line.strip().decode() for line in urllib.request.urlopen(url)]
    assert len(imagenet_classes) == NUM_IMAGENET_CLASSES
    print(f"Example ImageNet classes: {imagenet_classes[:10]}")

    # Tokenizer loader (we'll call it inside collate_fn as you wanted)
    tok_loader = _load_tok

    # Model
    config = {
        "img_feat_dim": 512,
        "text_feat_dim": 512,
        "shared_embedding_dim": 512,
    }
    model = CLIP(
        vocab_size=tok_loader(TOKENIZER_JSON_PATH, MAX_LEN).vocab_size,
        img_feat_dim=config["img_feat_dim"],
        text_feat_dim=config["text_feat_dim"],
        shared_dim=config["shared_embedding_dim"],
    )
    ckpt_path = os.path.join(f"./{run_name}_ckpts", checkpoint_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Dataset (streaming=True) and transforms
    imagenet_val = load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)

    val_transform = T.Compose([
        T.Lambda(to_pil_rgb),
        T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # Collate function that:
    #  - transforms images
    #  - builds 1000 prompts with a random template
    #  - re-tokenizes prompts on every batch
    def collate_fn(examples):
        # Images & labels from this batch
        imgs = [val_transform(ex["jpg"]) for ex in examples]
        images = torch.stack(imgs, dim=0)  # (B, 3, H, W)
        labels = torch.tensor([int(ex["cls"]) for ex in examples], dtype=torch.long)  # (B,)

        # Build prompts for ALL 1000 classes (class-ordered)
        prompts = build_random_template_prompts(imagenet_classes)

        # Re-load tokenizer (cached fast object) & re-tokenize this batch's prompts
        tok = tok_loader(TOKENIZER_JSON_PATH, MAX_LEN)
        enc = tok(prompts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        token_ids = enc["input_ids"].long()           # (1000, L)
        attention = enc["attention_mask"].long()      # (1000, L)
        lengths = attention.sum(dim=1).clamp_(min=1)  # (1000,)

        return images, token_ids, lengths, labels

    # DataLoader: batch of 1000 images; drop last to keep perfect 1000x1000 structure
    loader = DataLoader(
        imagenet_val,
        batch_size=NUM_IMAGENET_CLASSES,  # 1000 images per batch
        shuffle=False,                    # ignored for streaming IterableDataset
        drop_last=True,                   # ensures B == 1000 exactly (50 full batches for 50k val)
        num_workers=0,                    # safest for streaming eval
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    running_top1 = 0
    running_top5 = 0
    running_count = 0

    with torch.no_grad():
        for images, token_ids, lengths, labels in loader:
            # Move to device
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            token_ids = token_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward; your CLIP.forward returns: txt, img, logits_t, logits_i, scale
            _, _, _, logits_i, _ = model(images, token_ids, lengths)  # (B, 1000)

            # Top-k accuracies
            top5 = logits_i.topk(5, dim=-1).indices       # (B, 5)
            top1 = top5[:, 0]                              # (B,)
            running_top1 += (top1 == labels).sum().item()
            running_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            running_count += labels.size(0)

    print("Imagenet evaluation complete.")
    print(f"Top-1 Accuracy: {running_top1 / running_count:.4f}")
    print(f"Top-5 Accuracy: {running_top5 / running_count:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot ImageNet-1k evaluation with per-batch re-tokenization.")
    parser.add_argument("run_name", type=str, help="run name (used to find checkpoints under ./<run_name>_ckpts/)")
    parser.add_argument("checkpoint_name", type=str, help="checkpoint filename to load from disk")
    args = parser.parse_args()
    main(args.run_name, args.checkpoint_name)


