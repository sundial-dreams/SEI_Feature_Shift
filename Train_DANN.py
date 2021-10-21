import torch
from torch import nn, optim
from dataloader import loader
from tqdm import tqdm
import numpy as np
from DANNModel import DANNModel
from test import test
from helper import draw_table, draw_losses, draw_features, draw_confusion

use_gpu = torch.cuda.is_available()

loss_class = nn.NLLLoss()
loss_discriminator = nn.NLLLoss()

epoch = 50
batch_size = 128
lr = 0.001

source_dataloader, test_source_loader = loader("./dataset/dataset-frequency-std.mat", batch_size)
target_dataloader, test_target_loader = loader("./dataset/dataset-frequency2.mat", batch_size)
model = DANNModel(num_classes=4)

for p in model.parameters():
    p.requires_grad = True

optimizer = optim.Adam(params=model.parameters(), lr=lr)

if use_gpu:
    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_discriminator = loss_discriminator.cuda()

def train():
    model_losses = []
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    for e in range(epoch):
        total_loss = 0.0
        source_data_iter = iter(source_dataloader)
        target_data_iter = iter(target_dataloader)

        for i in tqdm(range(len_dataloader)):
            source_data = next(source_data_iter)
            x, y = source_data
            size = len(y)

            true_label = torch.ones(size).long()
            false_label = torch.zeros(size).long()

            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                true_label = true_label.cuda()
                false_label = false_label.cuda()

            p = float(i + e * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            class_out, domain_out = model(x, alpha)

            l_c = loss_class(class_out, y)
            l_d1 = loss_discriminator(domain_out, false_label)

            target_data = next(target_data_iter)

            x, _ = target_data

            if use_gpu:
                x = x.cuda()
            _, domain_out = model(x, alpha)

            l_d2 = loss_discriminator(domain_out, true_label)

            loss = l_c + l_d1 + l_d2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("epoch = {}, loss = {}".format(e, total_loss / len_dataloader))
        model_losses.append(total_loss / len_dataloader)

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
