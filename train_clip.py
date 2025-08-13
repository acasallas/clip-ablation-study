import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.transforms as transforms
import wandb


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")





# TODO: hey can you copy over final RNN and Alexnet from other file?
# also, can you look up datasets, keeping in mind max batch size for 4090? give yourself some room.
# assume you'll use tf32 and all that jazz.






# TODO: gonna need to tokenize, let's use huggingface.
# we'll have to download shakespeare and tokenize it.


# gotta change outputs.


# you should tokenize using something (sentencepice or tokenizer)
# step 1: train briefly on images, see that loss is going down.
# step 2: train briefly on text (maybe a sentiment analyzer?), see that loss is going down.

# TODO: strongly consider makin

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # then do CLIP
        self.embedding = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128,1) # takes last output and sigmoids it
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed) # we use the last hidden state here, but could've used output as well
        h_n = h_n.squeeze(0) # get rid of the time dimension
        out = self.fc(h_n)
        #return self.sigmoid(out) # let's not output prob, let's output logits
        return out


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv1 (valid padding)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # valid
            
            # Conv2 (same padding)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # valid
            
            # Conv3 (same)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv4 (same)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv5 (same)
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # valid
        )

        # Output after conv layers is: [batch_size, 256, 5, 5] → flatten to 6400
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x  # logits (no softmax)

# TODO: remove the classifier from AlexNET (or most of it I guess.)


# TODO: put hyperparameters for text_dim, img_dim, shared_dim.
class CLIP(nn.Module):
    def __init__(self, img_dim, text_dim, shared_dim):
        super().__init__()
        self.text_encoder = RNN(text_dim)
        self.image_encoder = AlexNet(img_dim)
        self.text_projector = nn.Linear(text_dim,shared_dim)
        self.image_projector = nn.Linear(img_dim,shared_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

        # initialize projectors for stability #TODO: look into this, is it a good idea?
        nn.init.normal_(self.text_projector.weight,  std=0.02)
        nn.init.normal_(self.image_projector.weight, std=0.02)
        nn.init.zeros_(self.text_projector.bias)
        nn.init.zeros_(self.image_projector.bias)

    def forward(self, x_img, x_text):
        encoded_text = self.text_encoder(x_text) # B, text_dim
        encoded_image = self.image_encoder(x_img) # B, img_dim
        projected_text = self.text_projector(encoded_text) # B , shared_dim
        projected_image = self.image_projector(encoded_image) # B, shared_dim
        norm_text = F.normalize(projected_text,p=2.0,dim=-1) # B , shared_dim
        norm_image = F.normalize(projected_image,p=2.0,dim=-1) # B, shared_dim
        logit_scale = torch.clamp(self.logit_scale.exp(), max=math.log(100))
        logits_per_text = (norm_text @ norm_image.transpose(0,1))*logit_scale # B,B
        logits_per_image = logits_per_text.t()
        return norm_text, norm_image, logits_per_text, logits_per_image

def clip_loss(logits_per_text, logits_per_image):
    B = logits_per_text.size(0)
    labels = torch.arange(B, device=logits_per_text.device)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    loss_i = F.cross_entropy(logits_per_image, labels)
    return 0.5 * (loss_t + loss_i)


def train_clip():
    model = CLIP() # TODO: put parameters in.
    model.to(device)

    for epoch in range(num_epochs):
        metrics = {"epoch": epoch}
        print(f"EPOCH {epoch}")
        train_losses = 0
        train_total = 0

        model.train()
        for i,data in enumerate(training_loader):
            images, texts = data
            # haven't made data yet, but when we do we'll guarantee images and text line up.
            inputs = inputs.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            _, _, logits_t, logits_i = model(images, texts)
            loss = clip_loss(logits_t, logits_i)

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


if __name__ == "__main__":
    train_clip()