import torch
from torch import nn
from models.GRL import ReverseLayerF
from models.resnet1d import resnet1d
use_gpu = torch.cuda.is_available()


class DANNModel(nn.Module):
    def __init__(self, num_classes: int):
        super(DANNModel, self).__init__()
        resnet = resnet1d(num_classes)
        self.size = 256
        # 特征提取
        if use_gpu:
            resnet = resnet.cuda()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

        self.resnet_extractor = resnet.feature_extract

        self.flatten = nn.AdaptiveAvgPool2d((1, 512))
        self.relu = nn.ReLU(True)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

        self.classify = nn.Sequential(
            nn.Linear(self.size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor, alpha: float):
        # x = x.view(x.size(0), 1, -1)
        x = self.resnet_extractor(x)
        # x = self.flatten(x)
        x = self.relu(x)
        # x = x.view(x.size(0), -1)
        reversed_x = ReverseLayerF.apply(x, alpha)

        class_out = self.classify(x)

        domain_out = self.domain_discriminator(reversed_x)
        return class_out, domain_out




