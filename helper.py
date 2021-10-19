import os
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
from typing import List
import json

def read(filename: str, start: int, end: int) -> list:
    r, k = [], 0
    with open(filename, "rb") as file:
        size = os.path.getsize(filename)
        while start > 0:
            file.read(4)
            start -= 1

        for i in range(size):
            if k > end:
                break
            data = file.read(4)

            if data == b'':
                continue
            value = struct.unpack('f', data)
            r.append(value[0])
            k += 1
    return r


def split_data(data: list, number: int) -> list:
    result = []
    data_size, i = len(data), 0
    while i < data_size:
        if i + number > data_size:
            break
        result.append(data[i: i + number])
        i += number
    return result


def file2dataset(filename: str, sample=1000, max_sample=2000) -> list:
    max_size = 1024 * max_sample  # 1000 sample with length is 1024
    data = read(filename, 0, max_size)
    dataset = split_data(data, 1024)
    np.random.shuffle(dataset)
    return dataset[:sample]


def dir2dataset(dirname: str) -> list:
    if os.path.isdir(dirname):
        result = []
        files = filter(lambda v: v.endswith(".bin"), os.listdir(dirname))
        for file in files:
            pattens = [r"(?<=mod)\d+", r"\w+(?=\_mod)", r"(?<=mod\d\_).+(?=\.bin)"]
            exps = map(lambda p: re.compile(p), pattens)
            filename = os.path.join(dirname, file)
            label, mod_type, freq = map(lambda e: e.findall(file)[0], exps)

            data = file2dataset(filename)
            # label - 1
            result.append([data, (int(label) - 1, mod_type, freq)])

        return result


def draw_losses(loss, title=""):

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss)
    plt.axis()
    plt.show()


def draw_confusion(confusion: torch.Tensor, title=""):
    fig, ax = plt.subplots()
    im = ax.matshow(confusion.numpy())
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xlabel="True",
        ylabel="Predict"
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def draw_features(features: torch.Tensor, labels: List[int]):
    if isinstance(features, torch.Tensor):
        fs = features.detach().numpy()
    else:
        fs = features
    for i in range(len(fs)):
        plt.plot(fs[i], label=labels[i])
    plt.legend()
    plt.show()

def draw_table(title: str, value: str, max_len=32):
    hint_str = "-" * max_len + "-"
    title_space_num = max_len - 14 - len(title)
    is_odd = title_space_num % 2 == 0
    title_space_num //= 2

    value_space_num = 10 - len(value)
    is_odd2 = value_space_num % 2 == 0

    value_space_num //= 2

    line_str = "|" + " " * title_space_num + title + " " * (title_space_num if is_odd else title_space_num + 1) + \
               "| " + value_space_num * " " + value + (value_space_num if is_odd2 else value_space_num + 1) * " " + " |"

    print(hint_str)
    print(line_str)
    print(hint_str)


def sec2min(sec: int) -> str:
    return str(format(sec / 60, ".1f")) + "min"

def save_json(filename, obj):
    with open(filename, "w") as f:
        json.dump(obj, f)
