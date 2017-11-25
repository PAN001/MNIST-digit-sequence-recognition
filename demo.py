import numpy as np

data_path = "./dataset/train_data_20_10000.npy"

data = np.load(data_path)
data = data / 255.0
np.save(data_path, data)