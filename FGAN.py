import torch
from torch import nn
from models.GRL import ReverseLayerF
from models.resnet1d import resnet1d

use_gpu = torch.cuda.is_available()

def reshape_feature(feature: torch.Tensor):
    return feature.view(feature.size(0), 1, -1)

class FGANModel(nn.Module):
    def __init__(self, num_classes, discriminator=None):
        super(FGANModel, self).__init__()
        self.size = 256
        self.split_size = 512
        resnet = resnet1d(num_classes)

        if use_gpu:
            resnet = resnet.cuda()

        self.extractor = resnet.feature_extract

        self.fc = nn.Sequential(
            nn.Linear(self.size, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
        )

        self.classify = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 64))
        )

        self.classify_fc = nn.Sequential(
            nn.Linear(64, 4),
            nn.LogSoftmax(dim=1)
        )

        self.discriminator = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 64))
        )

        self.discriminator_fc = nn.Sequential(
            nn.Linear(64, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor, alpha: float):
        x = self.extractor(x)
        x = self.fc(x)

        iml_feature, uml_feature = x[:, :self.split_size], x[:, self.split_size:]

        reversed_iml_feature, reversed_uml_feature = ReverseLayerF.apply(iml_feature, alpha), \
                                                     ReverseLayerF.apply(uml_feature, alpha)

        iml_feature = reshape_feature(iml_feature)
        uml_feature = reshape_feature(uml_feature)
        reversed_uml_feature = reshape_feature(reversed_uml_feature)
        reversed_iml_feature = reshape_feature(reversed_iml_feature)

        true_class_out = self.classify(uml_feature)
        true_class_out = self.classify_fc(true_class_out.view(true_class_out.size(0), -1))

        false_class_out = self.classify(reversed_iml_feature)
        false_class_out = self.classify_fc(false_class_out.view(false_class_out.size(0), -1))

        true_iml_out = self.discriminator(iml_feature)
        true_iml_out = self.discriminator_fc(true_iml_out.view(true_iml_out.size(0), -1))

        false_iml_out = self.discriminator(reversed_uml_feature)
        false_iml_out = self.discriminator_fc(false_iml_out.view(false_iml_out.size(0), -1))

        return true_class_out, false_class_out, true_iml_out, false_iml_out

