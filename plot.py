import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv

# plot best model
loss_path = "./saved_model_log/2scnn_1lstm_100_10000_9epoch/org_train_log.txt"
train_losses = {}
test_accs = []
cnt = 0
with open(loss_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    for row in csvReader:
        print "here"
        train_losses[int(row[0])] = (float(row[1]))
        cnt = cnt + 1

train_losses_list = []
keys = sorted(train_losses.keys())
for key in keys:
    train_losses_list.append(train_losses[key])
fig = plt.figure(2, figsize=(40, 10))
plt.plot(train_losses_list)
plt.title('2scnn_1lstm: model loss per batch')
# plt.yticks(range(0, 100))
plt.ylabel('loss')
plt.xlabel('batch')
# plt.legend(['Cifar10 train', 'Cifar10 test'], loc='upper left')
plt.show()
# plt.savefig("final_model_acc.png")
