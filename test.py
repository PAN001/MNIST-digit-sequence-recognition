import argparse
import torch
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', dest="model", type=str)
argparser.add_argument('--cuda', dest="cuda", type=bool, default=True)

args = argparser.parse_args()
net = torch.load(args.model)

dataset_data = np.load("./dataset/data_5_10000.npy")
dataset_data = dataset_data / 255.0
dataset_labels = np.load("./dataset/labels_5_10000.npy")
dataset_labels = dataset_labels.astype(int)