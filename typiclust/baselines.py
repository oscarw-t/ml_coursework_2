"""Uncertainty-based AL baselines: Uncertainty, Margin, Entropy.

Each strategy trains a ResNet18 on the current labeled set, computes softmax
responses on the unlabeled pool, then selects by the chosen score.
Cold start (empty labeled set) falls back to random selection, matching the
paper's treatment of L0 = empty (Appendix F.2.1).
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset


def uncertainty_select_round(labeled_indices, budget, n_total,
                              strategy='margin', device='cuda', epochs=100):
    """Select next batch using an uncertainty-based strategy.

    Args:
        strategy: 'uncertainty' (lowest max-prob),
                  'margin'      (lowest top-2 gap),
                  'entropy'     (highest entropy).
    """
    unlabeled = list(set(range(n_total)) - set(labeled_indices))

    if len(labeled_indices) == 0:
        # Cold start: no model available, fall back to random
        np.random.shuffle(unlabeled)
        return unlabeled[:budget]

    probs = _get_softmax_predictions(labeled_indices, unlabeled, device, epochs)

    if strategy == 'uncertainty':
        scores = -probs.max(axis=1)                        # lower max-prob -> higher score
    elif strategy == 'margin':
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        scores = -(sorted_p[:, 0] - sorted_p[:, 1])       # smaller margin -> higher score
    elif strategy == 'entropy':
        scores = -(probs * np.log(probs + 1e-10)).sum(axis=1)  # higher entropy -> higher score
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    top_local = np.argsort(-scores)[:budget]
    return [unlabeled[i] for i in top_local]


def _get_softmax_predictions(labeled_indices, unlabeled_indices, device, epochs):
    """Train ResNet18 on labeled set; return softmax probs (N_unlabeled, 10)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    infer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    infer_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=infer_transform
    )

    train_loader = DataLoader(Subset(full_train, labeled_indices),
                              batch_size=min(64, len(labeled_indices)),
                              shuffle=True, num_workers=2)
    infer_loader = DataLoader(Subset(infer_dataset, unlabeled_indices),
                              batch_size=256, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.025,
                                momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in infer_loader:
            logits = model(images.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(all_probs, axis=0)
