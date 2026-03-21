import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


@torch.no_grad()
def extract_features(model, device='cuda'):
    #extract L2 normalised 512 dim embeddings for all training images
    print("2. extracting features")

    plain_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=plain_transform
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    all_features, all_labels = [], []

    for images, labels in loader:
        features = model.get_features(images.to(device))
        features = F.normalize(features, dim=1)  # L2 normalise as per paper
        all_features.append(features.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0).numpy()  # (50000, 512)
    labels = torch.cat(all_labels, dim=0).numpy()
    print(f"  features: {features.shape}\n")
    return features, labels
