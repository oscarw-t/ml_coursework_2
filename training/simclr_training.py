import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.simclr_model import SimCLRModel


class SimCLRTransform:
    #applies the same random pipeline twice to get two different views
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class NTXentLoss(nn.Module):
    #z[i] and z[i+N] >= 0; all other 2N-2 pairs < 0
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        # z: (2N, dim), first N view1, last N view2
        N = z.shape[0] // 2
        z = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        labels = torch.cat([torch.arange(N, 2 * N),
                            torch.arange(0, N)]).to(z.device)

        # mask self-similarity so a point isn't its own negative
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)

        return F.cross_entropy(sim_matrix, labels)


def train_simclr(epochs=500, batch_size=512, lr=0.4, momentum=0.9,
                 weight_decay=1e-4, temperature=0.5, device='cuda',
                 checkpoint_path='simclr_checkpoint.'):
    """Train SimCLR on all 50k CIFAR-10 images (unlabelled).

    If checkpoint_path exists the model is loaded and returned immediately,
    skipping the 500-epoch training.  The checkpoint is saved after training.
    """
    model = SimCLRModel(feature_dim=128).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"1. Loading SimCLR checkpoint from '{checkpoint_path}'")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                          weights_only=True))
        return model

    print("1. SimCLR pre-training")

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=SimCLRTransform(size=32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NTXentLoss(temperature=temperature)

    model.train()
    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0

        for (view1, view2), _ in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            z = torch.cat([proj1, proj2], dim=0)  # (2N, 128)

            loss = criterion(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {total_loss/num_batches:.4f} "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
           

        if (epoch + 1) % 50 == 0 or epoch == 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Checkpoint saved to '{checkpoint_path}'")

    print("SimCLR training done.\n")
    return model
