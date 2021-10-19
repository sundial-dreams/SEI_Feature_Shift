import os
import struct
import numpy as np
import re
from scipy.io import loadmat, savemat
from sklearn.preprocessing import scale, normalize, MinMaxScaler


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


def dir2dataset(dirname: str, excludes=None) -> list:
    if excludes is None:
        excludes = {}
    if os.path.isdir(dirname):
        result = []
        files = filter(lambda v: v.endswith(".bin") and v not in excludes, os.listdir(dirname))
        for file in files:
            pattens = [r"(?<=mod)\d+", r"\w+(?=\_mod)", r"(?<=mod\d\_).+(?=\.bin)"]
            exps = map(lambda p: re.compile(p), pattens)
            filename = os.path.join(dirname, file)
            label, mod_type, freq = map(lambda e: e.findall(file)[0], exps)

            data = file2dataset(filename)
            # label - 1
            result.append([data, (int(label) - 1, mod_type, freq)])

        return result


def save_dataset(filename: str, dataset: list, key="dataset"):
    savemat(filename, {key: dataset})


def list_dataset(file_list: list):
    dataset = []
    for i in range(len(file_list)):
        data = file2dataset(file_list[i])
        for d in data:
            dataset.append([d, i])

    return dataset


def exclude_dataset(excludes: set):
    return dir2dataset("./raw_data", excludes=excludes)


def std_dataset(in_file, out_file, normal=True):
    dataset = loadmat(in_file)["dataset"]
    data = []
    label = []
    for d in dataset:
        data.append(d[0][0])
        label.append(d[1][0][0])
    data = np.array(data)
    label = label
    z_data1 = scale(data)
    if normal:
        z_data1 = MinMaxScaler().fit_transform(z_data1)
    dataset1 = []
    for i in range(len(z_data1)):
        dataset1.append([z_data1[i], label[i]])
    save_dataset(out_file, dataset1)


if __name__ == "__main__":
    # d = list_dataset(["./raw_data/un_mod1_2.4G.bin", "./raw_data/un_mod2_2.4G.bin", "./raw_data/un_mod3_2.4G.bin",
    #                   "./raw_data/un_mod4_2.4G.bin"])
    # save_dataset("./dataset/dataset-include[un_mod, 2.4G].mat", d)
    #
    # d = exclude_dataset({"./raw_data/un_mod1_2.4G.bin", "./raw_data/un_mod2_2.4G.bin", "./raw_data/un_mod3_2.4G.bin",
    #                      "./raw_data/un_mod4_2.4G.bin"})
    # save_dataset("./dataset/dataset-exclude[un_mod, 2.4G].mat", d)
    std_dataset("./dataset/dataset-exclude[un_mod, 2.4G].mat", "./dataset/dataset-exclude[un_mod, 2.4G]-std.mat", False)
    std_dataset("./dataset/dataset-include[un_mod, 2.4G].mat", "./dataset/dataset-include[un_mod, 2.4G]-std.mat", False)
