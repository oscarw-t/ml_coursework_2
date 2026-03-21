"""Framework (ii): linear classifier trained on frozen SimCLR features.

Per Appendix F.2.2:
  - Single linear layer of size d x C
  - lr = 0.025 * 100 = 2.5  (100x the supervised lr)
  - epochs = 2 * supervised epochs  (doubled training time)
  - SGD, momentum 0.9, Nesterov, cosine scheduler
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


@torch.no_grad()
def extract_test_features(model, device='cuda'):
    """L2-normalised 512-dim SimCLR features for the CIFAR-10 test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    feats, lbls = [], []
    for images, lbl in loader:
        f = model.get_features(images.to(device))
        feats.append(F.normalize(f, dim=1).cpu())
        lbls.append(lbl)

    return torch.cat(feats).numpy(), torch.cat(lbls).numpy()


def train_linear_probe(train_features, train_labels, test_features, test_labels,
                       device='cuda', supervised_epochs=100):
    """Train a linear head on frozen SimCLR features; return test accuracy (%).

    lr and epochs are scaled from the supervised values per Appendix F.2.2.
    """
    epochs = supervised_epochs * 2   # doubled
    lr = 0.025 * 100                 # 100x

    X_tr = torch.tensor(train_features, dtype=torch.float32)
    y_tr = torch.tensor(train_labels,   dtype=torch.long)
    X_te = torch.tensor(test_features,  dtype=torch.float32)
    y_te = torch.tensor(test_labels,    dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=min(64, len(y_tr)), shuffle=True)

    head = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    head.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = criterion(head(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    with torch.no_grad():
        preds = head(X_te.to(device)).argmax(dim=1).cpu()

    return (preds == y_te).float().mean().item() * 100.0
