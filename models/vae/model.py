# VAE fixed model with 3 convolutional layer, z = 180

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    	def __init__(self):
		super(VAE, self).__init__()
		self.eConv1 = nn.Conv2d(3,6,4)
		self.eConv2 = nn.Conv2d(6,12,5)
		self.ePool1 = nn.MaxPool2d(2,2)
		self.eConv3 = nn.Conv2d(12,24,5)
		self.ePool2 = nn.MaxPool2d(2,2)
		self.eF1 = nn.Linear(24*4*4,180)
		self.eMu = nn.Linear(180,180)
		self.eSigma = nn.Linear(180,180)

		self.dConvT1 = nn.ConvTranspose2d(180,200,4)
		self.dBatchNorm1 = nn.BatchNorm2d(200)
		self.dConvT2 = nn.ConvTranspose2d(200,120,6,2)
		self.dBatchNorm2 = nn.BatchNorm2d(120)
		self.dConvT3 = nn.ConvTranspose2d(120,60,6,2)
		self.dBatchNorm3 = nn.BatchNorm2d(60)
		self.dConvT4 = nn.ConvTranspose2d(60,3,5,1)

	def encode(self,x):
		x = self.eConv1(x)
		x = F.relu(x)
		x = self.eConv2(x)
		x = F.relu(x)
		x = self.ePool1(x)
		x = self.eConv3(x)
		x = F.relu(x)
		x = self.ePool2(x)
		x = x.view(x.size()[0], -1)
		x = self.eF1(x)
		mu = self.eMu(x)
		sigma = self.eSigma(x)
		return((mu,sigma))

	# From https://github.com/pytorch/examples/blob/master/vae/main.py
	def reparameterize(self,mu,sigma):
		std = torch.exp(0.5*sigma)
		eps = torch.randn_like(std)
		return (mu + eps*std)

	def decode(self,x):
		x = torch.reshape(x,(x.shape[0],180,1,1))
		x = self.dConvT1(x)
		x = self.dBatchNorm1(x)
		x = F.relu(x)
		x = self.dConvT2(x)
		x = self.dBatchNorm2(x)
		x = F.relu(x)
		x = self.dConvT3(x)
		x = self.dBatchNorm3(x)
		x = F.relu(x)
		x = self.dConvT4(x)
		x = torch.sigmoid(x)
		return(x)
		
	def forward(self,x):
		mu,sigma = self.encode(x)
		z = self.reparameterize(mu,sigma)
		x_gen = self.decode(z)
		return((x_gen,mu,sigma))