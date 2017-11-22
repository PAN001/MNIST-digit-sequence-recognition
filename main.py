import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from model import *
from CTCLoss import *

# Training settings
parser = argparse.ArgumentParser(description='Sequence MNIST Recognition')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate a pretrained model')
parser.add_argument('--model-path', type=str, default="model.pt", metavar='MP',
                    help='the path to the model to evaluate/save')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

print(args)

# set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# global variables
classes = 11
train_data_path = "./dataset/data_5_10000.npy"
train_labels_path = "./dataset/labels_5_10000.npy"

test_data_path = "./dataset/test_data_5_10000.npy"
test_labels_path = "./dataset/test_labels_5_10000.npy"

# load data
train_data = torch.Tensor(np.load(train_data_path))
train_labels = torch.IntTensor(np.load(train_labels_path).astype(int))
train_dataset = data_utils.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_data = torch.Tensor(np.load(test_data_path))
test_labels = torch.IntTensor(np.load(test_labels_path).astype(int))
test_dataset = data_utils.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

# initialize the model
if args.eval:
    model = torch.load(args.model_path)
else:
    model = Net(args.cuda, args.batch_size)

if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
# opt = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
criterion = CTCLoss(args.cuda)

def train(epoch):
    model.train()

    # reset states
    model.reset_hidden()
    model.reset_cell()
    model.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        out = model(data)

        out = out.view(args.batch_size, -1, classes)  # D(out) = (batch_size, seq_len, classes)
        out = out.permute(0, 2, 1)  # D(out) = (batch_size, classes, seq_len)

        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        # log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()

    # reset states
    model.reset_hidden()
    model.reset_cell()
    model.zero_grad()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        data, target = Variable(data, volatile=True), Variable(target)
        out = model(data)
        test_loss += criterion(out, target).data[0] # sum up batch loss

        out_np = out.data.cpu().numpy() if args.cuda else out.data.numpy()
        predictions = criterion.decode_best_path(out_np)

        print "best_path_predictions[0]: "
        print predictions[0]
        # print "best_path_predictions: "
        # print predictions

        # predictions_beam, scores_beam = criterion.decode_beam(out.data.numpy())
        # print "beam_predictions: "
        # print predictions_beam


        print "label[0]: "
        print target.data.cpu().numpy()[0] if args.cuda else target.data.numpy()[0]
        # print "label:"
        # print target.data.cpu().numpy() if cuda else target.data.numpy()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def save(filename):
    torch.save(model, filename)
    print('Model Saved as ', filename)

for epoch in range(1, args.epochs + 1):
    if args.eval:
        test()
    else:
        train(epoch)

if not args.eval:
    save(args.model_path)