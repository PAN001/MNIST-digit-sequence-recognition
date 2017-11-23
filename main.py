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
from Decoder import *
import os
import shutil
import sys

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # reset states
        batch_size = data.shape[0]
        model.reset_hidden(batch_size)
        model.reset_cell(batch_size)
        model.zero_grad()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        data, target = Variable(data), Variable(target)

        out = model(data)

        out = out.view(batch_size, -1, classes)  # D(out) = (batch_size, seq_len, classes)
        out = out.permute(0, 2, 1)  # D(out) = (batch_size, classes, seq_len)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

            out_np = out.data.cpu().numpy() if args.cuda else out.data.numpy()
            predictions, predictions_no_merge = decoder.decode_best_path(out_np)

            print "best_path_predictions_no_merge[0]: "
            print np.array(predictions_no_merge[0])

            print "best_path_predictions_merge[0]: "
            print np.array(predictions[0])

            print "label[0]: "
            print target.data.cpu().numpy()[0] if args.cuda else target.data.numpy()[0]

def validate():
    print "----------------------------------------Validation--------------------------------------------------"
    model.eval()

    validate_loss = 0.0
    validate_edit_dist = 0.0

    for data, target in validate_loader:
        # reset states
        batch_size = data.shape[0]
        model.reset_hidden(batch_size)
        model.reset_cell(batch_size)
        model.zero_grad()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        data, target = Variable(data, volatile=True), Variable(target)
        out = model(data)

        out = out.view(batch_size, -1, classes)  # D(out) = (batch_size, seq_len, classes)
        out = out.permute(0, 2, 1)  # D(out) = (batch_size, classes, seq_len)

        validate_loss += criterion(out, target).data[0] # sum up batch loss

        out_np = out.data.cpu().numpy() if args.cuda else out.data.numpy()
        target_np = target.data.cpu().numpy() if args.cuda else target.data.numpy()

        predictions, predictions_no_merge = decoder.decode_best_path(out_np)

        edit_dists, _, _, _, _ = decoder.edit_distance(target_np, predictions)
        # decoder.display_edit_diff(target_np, predictions)
        validate_edit_dist += sum(edit_dists)

        print "best_path_predictions_no_merge[0]: "
        print np.array(predictions_no_merge[0])

        print "best_path_predictions_merge[0]: "
        print np.array(predictions[0])

        # predictions_beam, scores_beam = criterion.decode_beam(out.data.numpy())
        # print "beam_predictions: "
        # print predictions_beam


        print "label[0]: "
        print target.data.cpu().numpy()[0] if args.cuda else target.data.numpy()[0]
        # print "label:"
        # print target.data.cpu().numpy() if cuda else target.data.numpy()



    validate_loss /= len(validate_loader.dataset) # average loss
    print "validate_edit_dist before averaging: ", validate_edit_dist
    validate_edit_dist /= float(len(validate_loader.dataset)) # average edit dist
    print('\nValidation set: Average loss: {:.4f}, Average edit dist: {:.4f}\n'.format(
        validate_loss, validate_edit_dist))

    print "----------------------------------------------------------------------------------------------------"
    return validate_edit_dist, validate_loss

def save(filename):
    torch.save(model, filename)
    print('Model Saved as ', filename)

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print "Update best model"
        shutil.copyfile(filename, 'model_best.pt') # update the best model: copy from filename to "model_best.pt"

def log(epoch, validate_edit_dist, validate_loss):
    with open(log_path, "a") as file:
        file.write(str(epoch) + "," + str(validate_edit_dist) + "," + str(validate_loss))


# Training settings
parser = argparse.ArgumentParser(description='Sequence MNIST Recognition')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--validate-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for validating (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
parser.add_argument('--model-path', type=str, default='', metavar='MP',
                    help='path to the model to evaluate/resume')

# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: None)')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

print(args)

# set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# global variables
start_epoch = 1
best_edit_dist = sys.maxint
validate_edit_dists = [] # for each epoch
validate_losses = [] # for each epoch

classes = 11

log_path = "./log.txt"
train_data_path = "./dataset/train_data_5_10000.npy"
train_labels_path = "./dataset/train_labels_5_10000.npy"

validate_data_path = "./dataset/test_data_5_1000.npy"
validate_labels_path = "./dataset/test_labels_5_1000.npy"

# load data
train_data = torch.Tensor(np.load(train_data_path) / 255.0)
train_labels = torch.IntTensor(np.load(train_labels_path).astype(int))
train_dataset = data_utils.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

validate_data = torch.Tensor(np.load(validate_data_path)[0:512] / 255.0)
validate_labels = torch.IntTensor(np.load(validate_labels_path).astype(int)[0:512])
validate_dataset = data_utils.TensorDataset(validate_data, validate_labels)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
    batch_size=args.validate_batch_size, shuffle=True, **kwargs)

# initialize the model
model = Net(args.cuda)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
# opt = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
criterion = CTCLoss(args.cuda)
decoder = Decoder()

# reload the model
if args.model_path:
    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        start_epoch = checkpoint['epoch']
        best_edit_dist = checkpoint['best_edit_dist']
        model.load_state_dict(checkpoint['state_dict']) # load model weights from the checkpoint
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))

if args.cuda:
    model.cuda()

# validate one test batch
if args.eval:
    validate()
    exit()

# train
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)

    # evaluate on validation set
    validate_edit_dist, validate_loss = validate()
    validate_edit_dists.append(validate_edit_dist)
    validate_losses.append(validate_loss)

    # remember best validate_edit_dist and save checkpoint
    is_best = validate_edit_dist <= best_edit_dist
    best_edit_dist = min(validate_edit_dist, best_edit_dist)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_edit_dist': best_edit_dist,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

    # log
    log(epoch, validate_edit_dist, validate_loss)
#
# if not args.eval:
#     save(args.model_path)