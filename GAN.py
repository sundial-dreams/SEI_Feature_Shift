import torch
from torch import nn
from torch.autograd import Function
from models.resnet1d import resnet1d


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GANModel(nn.Module):
    def __init__(self, num_classes: int, split_rate=0.5):
        super(GANModel, self).__init__()
        self.split_rate = 0.5
        self.size = 128
        # 特征提取
        self.extractor = resnet1d(num_classes).feature_extract
        # self.extractor = CNNModel().feature_extract
        # self.fc = nn.Linear(self.size, 512)
        self.relu = nn.ReLU(True)
        self.fc = nn.Sequential(
            nn.Linear(self.size, 512),
            # nn.ReLU(True),
            # nn.Linear(512, 2048),
            # nn.ReLU(True)
        )

        self.cnn_classifier = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 32)),
        )

        self.cnn_classifier_fc = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.cnn_discriminator = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 32)),
        )

        self.cnn_discriminator_fc = nn.Sequential(
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
        # 类别分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.size, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # 领域分类器
        self.discriminator = nn.Sequential(
            nn.Linear(self.size, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor, alpha):
        feature: torch.Tensor = self.extractor(x)
        feature = self.relu(feature)
        feature = self.fc(feature)

        uml_feature, iml_feature = feature[:, :self.size], feature[:, self.size:]
        # print(uml_feature.size(), iml_feature.size())
        reverse_iml_feature = ReverseLayerF.apply(iml_feature, alpha)

        uml_feature = uml_feature.view(uml_feature.size(0), 1, uml_feature.size(1))
        iml_feature = iml_feature.view(iml_feature.size(0), 1, iml_feature.size(1))
        reverse_iml_feature = reverse_iml_feature.view(reverse_iml_feature.size(0), 1, reverse_iml_feature.size(1))

        class_output = self.cnn_classifier(uml_feature)
        class_output = self.cnn_classifier_fc(class_output.view(class_output.size(0), -1))

        not_class_output = self.cnn_classifier(reverse_iml_feature)
        not_class_output = self.cnn_classifier_fc(not_class_output.view(not_class_output.size(0), -1))
        # 0 = uml, 1 = iml
        d_uml_out = self.cnn_discriminator(uml_feature)
        d_uml_out = self.cnn_discriminator_fc(d_uml_out.view(d_uml_out.size(0), -1))

        d_iml_out = self.cnn_discriminator(iml_feature)
        d_iml_out = self.cnn_discriminator_fc(d_iml_out.view(d_iml_out.size(0), -1))

        return class_output, not_class_output, d_uml_out, d_iml_out
