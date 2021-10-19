import torch
from torch import nn
from torch.utils.data import DataLoader

def test(model: nn.Module, test_loader: DataLoader, class_num=4) -> tuple:
    correct, count = 0, 0
    confusion = torch.zeros(class_num, class_num)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

            out, _, _, _ = model(x, 0)

            y_guess = torch.argmax(out, dim=1)
            correct += (y_guess == y).sum().item()
            count += len(x)

            for j in range(y_guess.size()[0]):
                confusion[y[j]][y_guess[j]] += 1

    for i in range(class_num):
        confusion[i] = confusion[i] / confusion[i].sum()

    return correct / count, confusion
