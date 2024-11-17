import torch
import wandb
import numpy as np
import torch.nn as nn


from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# Initialize wandb
wandb.init(
    project="AlexNet",
    config={
        "learning_rate": 0.005,
        "architecture": "AlexNet",
        "dataset": "CIFAR-10",
        "epochs": 20,
        "batch_size": 64,
        "weight_decay": 0.005,
        "momentum": 0.9,
    },
)

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Code running on {device}")


# Loading the dataset
def get_train_valid_loader(
    data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.203, 0.1994, 0.2010]
    )

    # define transforms
    valid_transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=valid_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


# CIFAR10 Dataset
train_loader, valid_loader = get_train_valid_loader(
    data_dir="data", batch_size=wandb.config.batch_size, augment=False, random_seed=42
)

test_loader = get_test_loader(data_dir="data", batch_size=wandb.config.batch_size)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(9216, 4096), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Initialize model, criterion, and optimizer
model = AlexNet(num_classes=10).to(device)
wandb.watch(model, log="all")  # Log model gradients and parameters

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=wandb.config.learning_rate,
    weight_decay=wandb.config.weight_decay,
    momentum=wandb.config.momentum,
)

# Training loop
total_step = len(train_loader)

for epoch in range(wandb.config.epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        train_loss += loss.item()

        # Log batch loss
        wandb.log({"batch_loss": loss.item(), "batch": epoch * len(train_loader) + i})

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{wandb.config.epochs}], "
                f"Step [{i+1}/{total_step}], "
                f"Loss: {loss.item():.4f}"
            )

    # Calculate average training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    # Calculate average validation metrics
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_accuracy = 100 * correct_valid / total_valid

    # Log epoch metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": avg_valid_loss,
            "valid_accuracy": valid_accuracy,
        }
    )

    print(
        f"Epoch [{epoch+1}/{wandb.config.epochs}] - "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.2f}%, "
        f"Valid Loss: {avg_valid_loss:.4f}, "
        f"Valid Acc: {valid_accuracy:.2f}%"
    )

# Save the model
torch.save(model.state_dict(), "alexnet_cifar10.pth")
wandb.save("alexnet_cifar10.pth")

# Close wandb run
wandb.finish()
