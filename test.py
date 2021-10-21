import torch
from torch import nn
from torch.utils.data import DataLoader

use_gpu = torch.cuda.is_available()


def test(model: nn.Module, test_loader: DataLoader, class_num=4, axis=0) -> tuple:
    correct, count = 0, 0
    confusion = torch.zeros(class_num, class_num)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if use_gpu:
                x = x.cuda()

            out = model(x, 0)
            out = out[axis]
            if use_gpu:
                out = out.cpu()

            y_guess = torch.argmax(out, dim=1)
            correct += (y_guess == y[axis]).sum().item()
            count += len(x)

            for j in range(y_guess.size()[0]):
                confusion[y[axis][j]][y_guess[j]] += 1

    for i in range(class_num):
        confusion[i] = confusion[i] / confusion[i].sum()

    return correct / count, confusion


# def test_freq()

