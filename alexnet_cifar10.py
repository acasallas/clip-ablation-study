import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
import wandb


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: valid
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0),  # 32→28
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 28→13 (valid)

            # Conv2: same
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 13→13
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 13→6 (valid)

            # Conv3–5: same
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),# 6→6
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),# 6→6
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),# 6→6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)                   # 6→2 (valid)
        )
        # After pools: [B, 256, 2, 2] → 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # logits
        )

    def forward(self, x):
        # x dim is B,C,H,W
        x = self.features(x)
        x = self.classifier(x) 
        return x # return dim should be B,num_classes


def plot_some_images(training_set):
    # Class labels
    classes = training_set.classes  # ['airplane', 'automobile', ..., 'truck']

    # Plot 5 images
    plt.figure(figsize=(10, 2))
    index_list = [45,234,1046,34056,45000]
    for i in range(len(index_list)):
        img, label = training_set[index_list[i]]  # img: tensor, label: int
        img = img.permute(1, 2, 0)  # change from [C,H,W] to [H,W,C] for matplotlib
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(classes[label])
        plt.axis("off")

    plt.show()


def main():
    config = {
        "weight_decay": 5e-4,
        "learning_rate": 0.001,
    }


    batch_size = 32
    num_epochs = 3

    # TODO: code review this.
    # TODO: put things on github (remove this comment before you do)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), # known means and stds for the three channels of the CIFAR10 dataset
                             (0.2470, 0.2435, 0.2616))
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), # known means and stds for the three channels of the CIFAR10 dataset
                             (0.2470, 0.2435, 0.2616))
        ])


    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
    validation_set = torchvision.datasets.CIFAR10('./data', train=False, transform=val_transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))


    # TODO: enable wandb
    with wandb.init(mode="disabled",config=config,project="alexnet-cifar10-playground",entity="alancasallas-self") as run:

        model = AlexNetCIFAR10(10)
        summary(model, input_size=(batch_size,3,32,32))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            metrics = {"epoch": epoch}
            print(f"EPOCH {epoch}")
            train_losses = 0
            train_total = 0
            train_correct = 0

            model.train()
            for i,data in enumerate(training_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = loss_fn(outputs,labels)
                train_losses += loss.item()*inputs.size(0)
                train_total += inputs.size(0)
                train_correct += (labels==predicted).sum().item()

                loss.backward()
                optimizer.step()
            print(f"train loss {train_losses/train_total:.4f} accuracy {train_correct/train_total:.4f}")
            metrics.update({"train_loss": train_losses/train_total, "train_accuracy": train_correct/train_total})

            model.eval()
            val_losses = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for i,data in enumerate(validation_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss =loss_fn(outputs,labels)
                    val_losses += loss.item()*inputs.size(0)
                    val_correct += (labels==predicted).sum().item()
                    val_total += labels.size(0)

            print(f"validation loss {val_losses/val_total:.4f} accuracy {val_correct/val_total:.4f}")
            metrics.update({"val_loss": val_losses/val_total, "val_accuracy": val_correct/val_total})
            wandb.log(metrics)



if __name__ == "__main__":
    main()
