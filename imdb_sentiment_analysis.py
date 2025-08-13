import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torchvision
import torchvision.transforms as transforms
import wandb

from transformers import PreTrainedTokenizerFast

from datasets import load_dataset


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_tokenizer():
    tokenizer_json = "imdb_bpe_bytelevel/tokenizer.json"

    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]"
    )


def load_tokenized_imdb_dataset(imdb_dataset, tokenizer):
    max_len = 256

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        # compute lengths from attention mask
        lengths = [int(sum(mask)) for mask in tokenized["attention_mask"]]
        tokenized["lengths"] = lengths
        # remove attention_mask if you don't need it
        del tokenized["attention_mask"]
        return tokenized

    tokenized = imdb_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # Now dataset has: input_ids, lengths, label
    tokenized.set_format(type="torch", columns=["input_ids", "lengths", "label"])
    return tokenized


# TODO: then let's check the max length to see what we're losing with max_len.


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128,1)
        #self.sigmoid = nn.Sigmoid() we'll output logits, not prob

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed) # we use the last hidden state here, but could've used output as well
        h_n = h_n.squeeze(0) # get rid of the time dimension
        out = self.fc(h_n)
        #return self.sigmoid(out) # let's not output prob, let's output logits
        return out

# a recommendation: If you observe overfitting, add a Dropout(0.3) before fc or set dropout on GRU if you use num_layers > 1.


def main():
    config = {
        "weight_decay": 1e-4,
        "learning_rate": 0.001,
        "embed_dim": 128
    }

    batch_size = 32
    num_epochs = 3

    print("Tokenizing IMDB dataset...")
    tokenizer = load_tokenizer()
    tokenized_dataset = load_tokenized_imdb_dataset(load_dataset("imdb"), tokenizer)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(tokenized_dataset["test"], batch_size=batch_size, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(tokenized_dataset["train"])))
    print('Validation set has {} instances'.format(len(tokenized_dataset["test"])))

    with wandb.init(mode="disabled", config=config,project="imdb-playground",entity="alancasallas-self") as run:

        model = RNN(len(tokenizer), wandb.config.embed_dim, tokenizer.pad_token_id)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # can we set a baseline evaluation run?

        for epoch in range(num_epochs):
            metrics = {"epoch": epoch}
            print(f"EPOCH {epoch}")
            train_losses = 0
            train_total = 0
            train_correct = 0

            model.train()
            for batch in training_loader:
                input_ids = batch["input_ids"].to(device)
                lengths   = batch["lengths"].to(device)
                labels    = batch["label"].float().to(device)

                optimizer.zero_grad()

                logits = model(input_ids, lengths).squeeze(1)
                loss = loss_fn(logits,labels)

                preds = (logits >= 0).long()
                train_losses += loss.item()*input_ids.size(0)
                train_total += input_ids.size(0)
                train_correct += (preds==labels).sum().item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print(f"train loss {train_losses/train_total:.4f} train accuracy {train_correct/train_total:.4f}")
            metrics.update({"train_loss": train_losses/train_total, "train_accuracy": train_correct/train_total})


            # remember model.train() and model.eval()!
            model.eval()
            val_losses = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in validation_loader:
                    input_ids = batch["input_ids"].to(device)
                    lengths   = batch["lengths"].to(device)
                    labels    = batch["label"].float().to(device)

                    logits = model(input_ids, lengths).squeeze(1)
                    preds = (logits >= 0).long()
                    loss = loss_fn(logits,labels)
                    val_losses += loss.item()*input_ids.size(0)
                    val_correct += (labels==preds).sum().item()
                    val_total += labels.size(0)

            print(f"validation loss {val_losses/val_total:.4f} accuracy {val_correct/val_total:.4f}")
            metrics.update({"val_loss": val_losses/val_total, "val_accuracy": val_correct/val_total})
            wandb.log(metrics)




if __name__ == "__main__":
    main()
