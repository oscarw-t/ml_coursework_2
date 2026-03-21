import torch.nn as nn
import torchvision.models as models


class SimCLRModel(nn.Module):
    #resNet18 encoder + MLP projection head
    #after train, use encoder features (512-dim), discard projections (128-dim)
    def __init__(self, feature_dim=128):
        super().__init__()

        resnet = models.resnet18(weights=None)

        #7x7 conv to 3x3, preserves spatial info on 32x32 CIFAR images
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        #remove maxpool, shrinks 32x32 too aggressively
        resnet.maxpool = nn.Identity()

        self.encoder_dim = resnet.fc.in_features  # 512
        resnet.fc = nn.Identity()
        self.encoder = resnet

        #512 to 512 to 128, used only during contrastive pte-training
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dim, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

    def get_features(self, x):
        return self.encoder(x)
