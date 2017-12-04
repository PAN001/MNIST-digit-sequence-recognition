import torch
from torchvision import datasets, transforms
import random
import numpy as np
from random import randint as rint
from scipy.misc import imsave
from PIL import Image
import os
import datetime
import random
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Generate Sequence MNIST Dataset')
parser.add_argument('--N', type=int, default=100, metavar='N',
                    help='number of digits in each sequence')
parser.add_argument('--M', type=int, default=1000, metavar='M',
                    help='number of samples')
parser.add_argument('--root-path', type=str, default='./dataset/', metavar='RP',
                    help='root path to the data to store')

args = parser.parse_args()
N = args.N # number of digits in the contiguous sequence
M = args.M # number of samples

# space = range(200, 10000)
# overlap = range(15, 25) # bigger -> more overlapped
space = range(200, 201)
overlap = range(15, 16) # bigger -> more overlapped

random.seed(123456789)

data = datasets.MNIST('./MNIST', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

ndata = torch.ByteTensor(np.lib.pad(data.train_data.numpy(), ((0,0),(4,4),(0,0)), 'constant'))

dataset_data = np.zeros((M, 36, 0))
dataset_labels = np.zeros((M, 36, 0))
s = np.append(data.train_labels.view(-1,1,1).repeat(1,36,1).numpy()[:M], ndata.numpy()[:M], axis=2)
for i in range(N):
    p = np.random.permutation(s)
    d = np.roll(p[:,:,1:], (0, rint(-4,4), 0), (0,1,2))
    if i == 0:
        dataset_data = d
    else:
        oq = rint(0, random.choice(overlap) - 9) + 9
        dd = np.append(np.zeros((M, 36, dataset_data.shape[2]-oq)), d, axis=2)
        dataset_data = np.append(dataset_data, np.zeros((M,36,28-oq)), axis=2)
        dataset_data += dd
    dataset_labels = np.append(dataset_labels, p[:,:,0:1], axis=2)

dataset_labels = dataset_labels[:,0,:]
# Creates a dataset of 60000 (28*N + (N-1)*overlap) * 36 images
# containing N numbers in sequence and their labels
images = []
if not os.path.exists('./images'): os.makedirs('./images')
for i in range(M):
    '''
    Randomly adding spacing bettween the numbers and then saving the images.
    '''
    img = np.zeros((36, 0))
    # probs = torch.Tensor(range(0, N + 1))

    dist = torch.multinomial(torch.ones(N+1), random.choice(space), replacement=True)
    for j in range(N+1):
        img = np.append(img, np.zeros((36, (dist==j).sum())), axis=1)
        img = np.append(img, dataset_data[i,:,28*j:28*(j+1)], axis=1)
    img = dataset_data[i,:,:]
    images.append(img)
    # name = './images/img_' + ''.join(map(lambda x: str(int(x)), dataset_labels[i])) + '.png'
    # imsave(name, img.clip(0, 255))

dataset_data = np.array(images) / 255.0

t = datetime.datetime.now().time()
if not os.path.exists(args.root_path): os.makedirs(args.root_path)
data_path = args.root_path + "data_" + str(N) + "_" + str(M) + ".npy"
np.save(data_path, dataset_data)
print("Saved: ", data_path)
label_path = args.root_path + "labels_" + str(N) + "_" + str(M) + ".npy"
np.save(label_path, dataset_labels)
print("Saved: ", label_path)


# dist = torch.multinomial(torch.ones(N+1), space, replacement=True)