from torch import from_numpy
import numpy as np
import argparse
from os import path

def load_data(data_path):
    """
    Load the data from the specified directory.

    :param data_path: Path to the data.
    :return: List of train, dev and test loaders.
    """
    with open("../../config/seeds.txt", "r") as fp:
        seeds = fp.read().splitlines()

    print(f"Loading data from {data_path}...")
    for i in range(len(seeds)):
        train_X = from_numpy(np.load(path.join(data_path, f"split_{i}", "train_X.npy")))
        train_mask = from_numpy(np.load(path.join(data_path, f"split_{i}", "train_mask.npy")))
        train_y = from_numpy(np.load(path.join(data_path, f"split_{i}", "train_y.npy"))).float()

        assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

        dev_X = from_numpy(np.load(path.join(data_path, f"split_{i}", "dev_X.npy")))
        dev_mask = from_numpy(np.load(path.join(data_path, f"split_{i}", "dev_mask.npy")))
        dev_y = from_numpy(np.load(path.join(data_path, f"split_{i}", "dev_y.npy"))).float()

        assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]

        print("Loading test data from directory {}...".format(path.join(data_path, "split_{}".format(i))))
        test_X = from_numpy(np.load(path.join(data_path, f"split_{i}", "test_X.npy")))
        test_mask = from_numpy(np.load(path.join(data_path, f"split_{i}", "test_mask.npy")))
        test_y = from_numpy(np.load(path.join(data_path, f"split_{i}", "test_y.npy"))).float()

        assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

        print(f"{i} - Train: {train_X.shape[0]}, Dev: {dev_X.shape[0]}, Test: {test_X.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./", help="Path to the data.")
    args = parser.parse_args()

    load_data(args.data_path)