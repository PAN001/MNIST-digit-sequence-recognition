import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv

def read_in(loss_path):
    # loss_path = "./saved_model_log/2scnn_1lstm_100_10000_9epoch/org_train_log.txt"
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

    # return train_losses_list
    return train_losses_list

def read_validation(loss_path):
    val_losses = {}
    val_lers = {}
    cnt = 0
    with open(loss_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            print "here"
            val_losses[int(row[0])] = (float(row[2]))
            val_lers[int(row[0])] = (float(row[1]))
            cnt = cnt + 1

    val_losses_list = []
    val_lers_list = []
    keys = sorted(val_losses.keys())
    for key in keys:
        val_losses_list.append(val_losses[key])
        val_lers_list.append(val_lers[key])

    # return train_losses_list
    return val_losses_list, val_lers_list

# # 6 models loss vs. batch
# train_losses_list1 = read_in("./saved_model_log/1-2lcnn_1lstm_100_10000_31epoch/1lstm_2cnn_100_train_log.txt");
# train_losses_list2 = read_in("./saved_model_log/2-2lcnn_1bilstm_100_10000_30epoch/1bilstm_2cnn_100_train_log.txt");
# train_losses_list3 = read_in("./saved_model_log/3-2lcnn_2bilstm_100_10000_27epoch/2bilstm_2cnn_100_train_log.txt");
# train_losses_list4 = read_in("./saved_model_log/4-2scnn_1lstm_100_10000_9epoch/org_train_log.txt");
# train_losses_list5 = read_in("./saved_model_log/5-2scnn_2bilstm_100_10000_29epoch/2scnn_2bilstm_train_log.txt");
# train_losses_list6 = read_in("./saved_model_log/6-2scnn_2bilstm_scaled_10000_59epoch/2scnn_2bilstm_scaled_100_train_log.txt");
#
#
# fig = plt.figure(2, figsize=(40, 10))
# # x = range(300)[0:-1:5]
# plt.plot(train_losses_list1)
# plt.plot(train_losses_list2)
# plt.plot(train_losses_list3)
# plt.plot(train_losses_list4)
# plt.plot(train_losses_list5)
# plt.plot(train_losses_list6)
# max_len = max(len(train_losses_list1), len(train_losses_list2), len(train_losses_list3), len(train_losses_list4), len(train_losses_list5), len(train_losses_list6))
# plt.title('MNIST Sequence Recognition: loss vs. batch (batch size = 16)')
# axes = plt.gca()
# axes.set_ylim([0, 2500])
# # axes.set_xlim([500, 2500])
# # axes.set_ylim([0, 3000])
# # axes.set_xlim([0, 20000])
# # axes.set_xlim([100, 200])
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['model-1', 'model-2', 'model-3', 'model-4', 'model-5', 'model-6'], loc='upper left')
# plt.xticks([0,100,200,300,400], ["500", "1500", "2500", "3500", "4500"])
# plt.show()
# # plt.savefig("./plots/loss.png")

# # model-6 train
# train_losses_list6 = read_in("./saved_model_log/6-2scnn_2bilstm_scaled_10000_59epoch/2scnn_2bilstm_scaled_100_train_log.txt");
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(train_losses_list6)
# plt.title('Model-6 Training: loss vs. batch (batch size = 16)', fontsize=28)
# plt.ylabel('loss', fontsize=18)
# plt.xlabel('batch', fontsize=18)
# axes = plt.gca()
# axes.set_ylim([0, 600])
# # axes.set_xlim([500, 2500])
# # plt.legend(['model-1', 'model-2', 'model-3', 'model-4', 'model-5', 'model-6'], loc='upper left')
# plt.xticks([0,500,1000,1500,2000,2500,3000,3500], ["0", "5000", "10000", "15000", "20000", "25000", "30000", "35000"])
# # plt.show()
# plt.savefig("./plots/model-6_train_loss.png")

# model-6 validation
val_losses_list6, val_lers_list6 = read_validation("./saved_model_log/6-2scnn_2bilstm_scaled_10000_59epoch/2scnn_2bilstm_scaled_100_validation_log.txt");
# # LER
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(val_lers_list6)
# plt.title('Model-6 Validation: LER vs. epoch', fontsize=28)
# plt.ylabel('LER(%)', fontsize=18)
# plt.xlabel('batch', fontsize=18)
# axes = plt.gca()
# # axes.set_ylim([0, 600])
# # axes.set_xlim([500, 2500])
# # plt.legend(['model-1', 'model-2', 'model-3', 'model-4', 'model-5', 'model-6'], loc='upper left')
# # plt.xticks([0,500,1000,1500,2000,2500,3000,3500], ["0", "5000", "10000", "15000", "20000", "25000", "30000", "35000"])
# # plt.show()
# plt.savefig("./plots/model-6_validation_ler.png")
# loss
fig = plt.figure(2, figsize=(40, 10))
plt.plot(val_losses_list6)
plt.title('Model-6 Validation: loss vs. epoch', fontsize=28)
plt.ylabel('loss', fontsize=18)
plt.xlabel('batch', fontsize=18)
axes = plt.gca()
# axes.set_ylim([0, 600])
# axes.set_xlim([500, 2500])
# plt.legend(['model-1', 'model-2', 'model-3', 'model-4', 'model-5', 'model-6'], loc='upper left')
# plt.xticks([0,500,1000,1500,2000,2500,3000,3500], ["0", "5000", "10000", "15000", "20000", "25000", "30000", "35000"])
# plt.show()
plt.savefig("./plots/model-6_validation_loss.png")