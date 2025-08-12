import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import wandb


def load_tokenized_imdb_set():
    pass


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # then do CLIP
        self.embedding = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128,1)
        #self.sigmoid = nn.Sigmoid() we'll output logits, not prob

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed) # we use the last hidden state here, but could've used output as well
        h_n = h_n.squeeze(0) # get rid of the time dimension
        out = self.fc(h_n)
        #return self.sigmoid(out) # let's not output prob, let's output logits
        return out






def main():
    config = {
        "weight_decay": 0.1,
        "learning_rate": 0.001,
        "neuron_size": 200
    }


    batch_size = 16
    num_epochs = 3


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    # can we download the finewebedu set, a little sample?
    # define dataloader here
    print(f'Training set has shape {training_set[0][0].shape}')

    with wandb.init(config=config,project="shakespeare-playground",entity="alancasallas-self") as run:
        # first step: turn tokens into embeddings
        # second step: get a positional embeddings layer and add it.
        # then, do attention with causal masking.
        # the important part is attention part, you can use closer guidance for the residual and init part.


        # after mnist, ya gotta move on to an actual transformer today, implement embedding and attention.

        model = MLP(wandb.config.neuron_size)
        model.to(device)

        # define training loop here
        # todo: can you implement batch accumulation?

        # let's try Adam for now with default parameters
        # hyperparameters, can we get them into wandb?
        optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            metrics = {"epoch": epoch}
            print(f"EPOCH {epoch}")
            train_losses = 0
            train_total = 0

            model.train()
            for i,data in enumerate(training_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs,labels)
                train_losses += loss.item()*inputs.size(0)
                train_total += inputs.size(0)
                loss.backward()
                optimizer.step()
            print(f"train loss {train_losses/train_total}")
            metrics.update({"train_loss": train_losses/train_total})


            # remember model.train() and model.eval()!
            model.eval()
            val_losses = 0
            val_correct = 0
            val_total = 0

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
            wandb.log(metrics)




if __name__ == "__main__":
    main()
