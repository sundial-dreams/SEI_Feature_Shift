import torch
from torch import nn, optim
from dataloader import loader
from tqdm import tqdm
import numpy as np
from FGAN import FGANModel
from test import test
from helper import draw_table, draw_losses, draw_features, draw_confusion

use_gpu = torch.cuda.is_available()

ture_class_loss_fn = nn.NLLLoss()
false_class_loss_fn = nn.NLLLoss()
true_iml_loss_fn = nn.NLLLoss()
false_iml_loss_fn = nn.NLLLoss()

epoch = 30
batch_size = 128
lr = 0.001

source_dataloader, test_source_loader = loader("./dataset/dataset-frequency-un_mod-all_freq-std.mat", batch_size)
target_dataloader, test_target_loader = loader("./dataset/dataset-frequency-2.4G-std.mat", batch_size)

model = FGANModel(num_classes=4)

freq_map = {
    "1G": 0, "1.5G": 1, "2.45G": 2, "2.5G": 3
}

reversed_freq_map = ["1G", "1.5G", "2.45G", "2.5G"]

for p in model.parameters():
    p.requires_grad = True

optimizer = optim.Adam(params=model.parameters(), lr=lr)

if use_gpu:
    model = model.cuda()
    false_iml_loss_fn = false_iml_loss_fn.cuda()
    ture_class_loss_fn = ture_class_loss_fn.cuda()
    true_iml_loss_fn = true_iml_loss_fn.cuda()
    false_class_loss_fn = false_class_loss_fn.cuda()


def train():
    model_losses = []
    len_dataloader = len(source_dataloader)
    for e in range(epoch):
        total_loss = 0.0
        for i, (x, y) in tqdm(enumerate(source_dataloader), total=len_dataloader):
            # size = len(y)

            if use_gpu:
                x = x.cuda()
                y = torch.stack(y, dim=0).cuda()

            p = float(i + e * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # print(alpha)
            optimizer.zero_grad()

            class_out, not_class_out, d_uml_out, d_iml_out = model(x, alpha)

            loss = 0.5 * ture_class_loss_fn(class_out, y[0]) + false_class_loss_fn(not_class_out, y[0]) + \
                   0.5 * true_iml_loss_fn(d_iml_out, y[2]) + false_iml_loss_fn(d_uml_out, y[2])

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

    accuracy, confusion = test(model, test_source_loader, axis=2)
    draw_table("Test Accuracy", format(accuracy, ".4f"))
    draw_confusion(confusion)

    print("-" * 10, "target dataset", "-" * 10)
    accuracy, confusion = test(model, test_target_loader)
    draw_table("Test Accuracy", format(accuracy, ".4f"))
    draw_confusion(confusion)

    torch.save(model, "./model_saved/FGAN_model.pt")
