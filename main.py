import torch
from torch import nn, optim
from dataloader import loader
from tqdm import tqdm
import numpy as np
from GAN import GANModel
from test import test
from helper import draw_table, draw_losses, draw_features, draw_confusion

loss_class = nn.NLLLoss()
loss_not_class = nn.NLLLoss()
loss_n_d = nn.NLLLoss()
loss_d = nn.NLLLoss()

epoch = 20
batch_size = 32
lr = 0.001

source_dataloader, test_source_loader = loader("./dataset/dataset-frequency-std.mat", batch_size)
target_dataloader, test_target_loader = loader("./dataset/dataset-frequency2-std.mat", batch_size)

model = GANModel(num_classes=4)

for p in model.parameters():
    p.requires_grad = True

optimizer = optim.Adam(params=model.parameters(), lr=lr)


def train():
    model_losses = []
    len_dataloader = len(source_dataloader)
    for e in range(epoch):
        for i, (x, y) in tqdm(enumerate(source_dataloader), total=len_dataloader):
            p = float(i + e * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            class_out, not_class_out, d_uml_out, d_iml_out = model(x, alpha)

            l_c = loss_class(class_out, y)
            l_nc = loss_class(not_class_out, y)

            size = len(y)

            l_d = 0.5 * (loss_n_d(d_iml_out, torch.zeros(size).long()) + loss_d(d_uml_out, torch.ones(size).long()))
            loss = l_c + l_nc + l_d

            loss.backward()
            optimizer.step()
        print("epoch = {}, loss = {}".format(e, loss.item()))
        model_losses.append(loss.item())

    return model_losses


if __name__ == "__main__":
    losses = train()

    draw_losses(losses)

    accuracy, confusion = test(model, test_source_loader)
    draw_table("Test Accuracy", format(accuracy, ".4f"))
    draw_confusion(confusion)

    print("-" * 10, "target dataset", "-" * 10)
    accuracy, confusion = test(model, test_target_loader)
    draw_table("Test Accuracy", format(accuracy, ".4f"))
    draw_confusion(confusion)
