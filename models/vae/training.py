from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import datasets, transforms
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CIFAR10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = args.batch_size


# Data loaders for training and testing
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

# load model
vae = model.to(device)

optimizer = optim.Adam(vae.parameters(), lr= args.lr)

def loss_function(x, x_gen, mu, sigma):
	BCE = F.binary_cross_entropy(x_gen, x, reduction='sum')
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

	return BCE + KLD

# training
for epoch in range(args.epochs):
    
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		images = data[0].to(device)
		#images = data[0]
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = vae(images)
		loss = loss_function(images, outputs[0], outputs[1], outputs[2])
		loss.backward()
		optimizer.step()

		# print statistics
	running_loss += loss.item()
	# print loss every epoch
	print('epoch %d loss: %.3f' %(epoch + 1, running_loss / len(trainloader)))
	running_loss = 0.0

	if epoch % args.log_interval == (args.log_interval-1):
        # every 100 th epoch save the checkpoint
        PATH = './vae_checkpoints/'
	    torch.save(vae.state_dict(), PATH+str(epoch+1)+".pt")
        
print('Finished Training')

def main():
    	# Training settings
	parser = argparse.ArgumentParser(description='Training a VAE model')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)

if __name__ == '__main__':
	main()