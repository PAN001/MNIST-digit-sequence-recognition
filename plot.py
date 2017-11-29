import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv

def read_in(loss_path):
    # loss_path = "./saved_model_log/2scnn_1lstm_100_10000_9epoch/org_train_log.txt"
    train_losses_list = []
    train_losses = {}
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

    max_len = 5000
    if len(train_losses_list) < max_len:
        max_len = len(train_losses_list)

    return train_losses_list
    # return train_losses_list[500:max_len]

train_losses_list1 = read_in("./saved_model_log/1-2lcnn_1lstm_100_10000_31epoch/1lstm_2cnn_100_train_log.txt");
train_losses_list2 = read_in("./saved_model_log/2-2lcnn_1bilstm_100_10000_30epoch/1bilstm_2cnn_100_train_log.txt");
train_losses_list3 = read_in("./saved_model_log/3-2lcnn_2bilstm_100_10000_27epoch/2bilstm_2cnn_100_train_log.txt");
train_losses_list4 = read_in("./saved_model_log/4-2scnn_1lstm_100_10000_9epoch/org_train_log.txt");
train_losses_list5 = read_in("./saved_model_log/5-2scnn_2bilstm_100_10000_29epoch/2scnn_2bilstm_train_log.txt");
train_losses_list6 = read_in("./saved_model_log/6-2scnn_2bilstm_scaled_10000_59epoch/2scnn_2bilstm_scaled_100_train_log.txt");


fig = plt.figure(2, figsize=(40, 10))
plt.plot(train_losses_list1)
plt.plot(train_losses_list2)
plt.plot(train_losses_list3)
plt.plot(train_losses_list4)
plt.plot(train_losses_list5)
plt.plot(train_losses_list6)
max_len = max(len(train_losses_list1), len(train_losses_list2), len(train_losses_list3), len(train_losses_list4), len(train_losses_list5), len(train_losses_list6))
plt.title('plot')
# plt.xticks([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000])
axes = plt.gca()
axes.set_ylim([0, 1000])
axes.set_xlim([500, 2500])
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['model-1', 'model-2', 'model-3', 'model-4', 'model-5', 'model-6'], loc='upper left')
plt.show()
# plt.savefig("final_model_acc.png")


