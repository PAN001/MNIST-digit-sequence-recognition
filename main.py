import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from CTCLoss import *
from Decoder import *
import os
import shutil
import sys
import time

# import model
# from model_2scnn_2bilstm import *
from model_2scnn_2bilstm_scaled import *
# from inception_2bilstm import *

def train(epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
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

        # print "data shape: ", data.size()

        out = model(data)

        out = out.view(batch_size, -1, classes)  # D(out) = (batch_size, seq_len, classes)
        # print out.size()
        out = out.permute(0, 2, 1)  # D(out) = (batch_size, classes, seq_len)

        loss = criterion(out, target)
        losses.update(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # standard output
        if batch_idx % args.log_interval == 0:
            print args.id
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss = losses, batch_time = batch_time))

            out_np = out.data.cpu().numpy() if args.cuda else out.data.numpy()
            test_out_np = out_np[0].reshape(1, out_np[0].shape[0], out_np[0].shape[1])
            predictions, predictions_no_merge = decoder.decode_best_path(test_out_np)

            print "best_path_predictions_no_merge[0]: "
            print np.array(predictions_no_merge[0])

            print "best_path_predictions_merge[0]: "
            print np.array(predictions[0])

            print "label[0]: "
            print target.data.cpu().numpy()[0] if args.cuda else target.data.numpy()[0]

            target_np = target.data.cpu().numpy() if args.cuda else target.data.numpy()
            test_target_np = target_np[0].reshape(1, -1)

            edit_dists, _, _, _, _ = decoder.edit_distance(test_target_np, predictions)
            print "Edit distance is: ", edit_dists[0]

            print ""

            # log
            log((epoch - 1) * (training_num / args.batch_size) + batch_idx, losses.val, train_log_path)

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
    if is_best or state["epoch"] < 20:
        print "=> Update best model to: ", best_model_path
        shutil.copyfile(filename, best_model_path) # update the best model: copy from filename to "model_best.pt"

def log(epoch, validate_loss, log_path, validate_edit_dist = None):
    with open(log_path, "a") as file:
        if validate_edit_dist != None:
            file.write(str(epoch) + "," + str(validate_edit_dist) + "," + str(validate_loss) + "\n")
        else:
            file.write(str(epoch) + "," + str(validate_loss) + "\n")

    print "=> Logged"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Training settings
parser = argparse.ArgumentParser(description='Sequence MNIST Recognition')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--validate-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for validating (default: 256)')
parser.add_argument('--epoch', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate a pretrained model')
parser.add_argument('--model-path', type=str, default='', metavar='MP',
                    help='path to the model to evaluate/resume')
parser.add_argument('--id', type=str, default='null', metavar='ID',
                    help='id of each training instance')
parser.add_argument('--train-len', type=str, default='100', metavar='TRLEN',
                    help='number of digits in each sequence image (training)')
parser.add_argument('--test-len', type=str, default='100', metavar='TELEN',
                    help='number of digits in each sequence image (testing)')

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
best_model_path = "./" + args.id +  "_best_model.pt"
print "best_model_path: ", best_model_path

classes = 11

train_log_path = "./" + args.id + "_train_log.txt"
print "train_log_path: ", train_log_path
validation_log_path = "./" + args.id + "_validation_log.txt"
print "validation_log_path: ", validation_log_path

train_data_path = "./dataset/data_" + args.train_len + "_10000_random.npy"
train_labels_path = "./dataset/labels_" + args.train_len + "_10000_random.npy"

# validate_data_path = "./dataset/test_data_" + args.test_len + "_1000.npy"
# validate_labels_path = "./dataset/test_labels_" + args.test_len + "_1000.npy"

validate_data_path = "./dataset/test_data_" + args.test_len + "_sun.npy"
validate_labels_path = "./dataset/test_labels_" + args.test_len + "_sun.npy"


# load data
if not args.eval:
    print "=> Loading train data: ", train_data_path
    train_data = np.load(train_data_path)
    training_num = train_data.shape[0]
    train_data = torch.Tensor(train_data)
    train_labels = torch.IntTensor(np.load(train_labels_path).astype(int))
    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print "=> Loaded train data: ", train_data_path

print "=> Loading validation data: ", validate_data_path
validate_data = torch.Tensor(np.load(validate_data_path))
validate_labels = torch.IntTensor(np.load(validate_labels_path).astype(int))
validate_dataset = data_utils.TensorDataset(validate_data, validate_labels)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
    batch_size=args.validate_batch_size, shuffle=True, **kwargs)
print "=> Loaded validation data: ", validate_data_path

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

# print the model architecture
print model

# validate one test batch
if args.eval:
    validate()
    exit()

# train
for epoch in range(start_epoch, args.epoch + 1):
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
    log(epoch, validate_loss, validation_log_path, validate_edit_dist)
#
# if not args.eval:
#     save(args.model_path)