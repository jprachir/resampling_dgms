import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
# from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image



parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=100,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0004, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=1, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-D', '--debug', action='store_true', help='debug mode: does not save the model')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# MODEL NAME
model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
assert args.debug or not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
model_name = 'test' if args.debug else model_name

# MISCLLANEOUS
sample_batch_size = 500
obs = (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':8, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

# DATA
if 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

# MODEL NAME
model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = nn.DataParallel(model)
model = model.cuda()
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# LOAD params
# if args.load_params:
    # load_part_of_model(model, args.load_params)
path = "/content/gdrive/MyDrive/deepak/pixel-cnn-pp/models/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"
model.load_state_dict(torch.load(path))
print('model parameters loaded')

# SAMPLE UTILITY
@torch.no_grad()
def sample(model):
    model.train(False)
    model.cuda()
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]): # width
        for j in range(obs[2]): #height
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            # print("model returning shape ", out.shape) # [25,100,32,32]
            out_sample = sample_op(out) 
            # print("out sample shape ", out_sample.shape) # [25,3,32,32]
            data[:, :, i, j] = out_sample.data[:, :, i, j]
            # print("data shape ",data.shape) # [25,3,32,32]
    return data

list_of_img_arr = []
for i in range(90):
    print('iteration -->  ', i)
    sample_t = sample(model)
    # print("***function sample shape ",sample_t.shape,"***") #[25,3,32,32]
    sample_t = rescaling_inv(sample_t)
    # print(sample_t.shape, type(sample_t)) # [25,3,32,32]
    # transforms.ToPILImage()(sample_t.squeeze(0))
    utils.save_image(sample_t[0],'sample_1.png')
    utils.save_image(sample_t[1],'sample_2.png')
    # utils.save_image(sample_t,'new_samples/{}_{}.png'.format(model_name, sample_t.size(0)),
            # nrow=5, padding=0) 
    batch_250 = sample_t.cpu().permute(0,2,3,1).numpy()
    # print("-----",type(batch_25_np)) # np -> [1,32,32,3] 
    # 250 numpy images
    list_of_img_arr.append(batch_250)
  
    #for 20 epochs (5000array)
    if i % 10 == 0:
        # np.savez_compressed("./all_train_batches_5k/batch_{}.npz".format(i), batch_250)
        np.savez_compressed('./train_generation.npz', train_data = np.vstack(list_of_img_arr))
        
# saves training data
np.savez_compressed('./train_generation.npz', train_data = np.vstack(list_of_img_arr)) #[45k,32,32,3]

# work
'''
path = "/content/gdrive/MyDrive/deepak/pixel-cnn-pp/pixel-cnn-pp/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"
pytorch_model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
pytorch_model.load_state_dict(torch.load(path),strict=False)
pytorch_model = nn.DataParallel(pytorch_model)
sample_t = sample(pytorch_model)
sample_t = rescaling_inv(sample_t)
utils.save_image(sample_t, 'images/my_sample.png', nrow=5, padding=0)
'''

