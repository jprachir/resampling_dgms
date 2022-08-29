import torch
import numpy as np
import os

path = "/content/gdrive/MyDrive/deepak/lady_vae/xgen"

train_data = np.load(os.path.join(path, 'train_data.npz'))['train_data']
valid_data = np.load(os.path.join(path, 'valid_data.npz'))['valid_data']

ar1 = np.load(os.path.join(path, 'test_data_1.npz'))['test_data']
ar2 = np.load(os.path.join(path, 'test_data_2.npz'))['test_data_2nd_half']
test_data = np.vstack((ar1,ar2))

np.savez_compressed('gen_samples.npz', train_data = train_data, valid_data = valid_data,
                    test_data = test_data)


