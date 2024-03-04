import os
import numpy as np


def create_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters_in_MB(model):
    return (
        sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary_head" not in name
        )
        / 1e6
    )


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_cifar100_classlist(root):
    import os

    meta = os.path.join(root, "cifar-100-python", "meta")
    meta = unpickle(meta)
    classlist = meta[b"fine_label_names"]
    return list(map(lambda _: _.decode(), classlist))
