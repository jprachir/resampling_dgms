### Primary Reference:
SOTA DGMs can exhibit imperfections e.g. image models can have noticeable artifacts in the background. The source of such bias is due to mismatch in the base generative model family and the true data distribution, or simply challenges in optimization. This is a framework can characterize and mitigate the model bias.

> Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting  
> [Aditya Grover](https://aditya-grover.github.io), Jiaming Song, Alekh Agarwal, Kenneth Tran, Ashish Kapoor, Eric Horvitz, Stefano Ermon.  
> Paper: https://arxiv.org/abs/1906.09531  

### Differences with official implementation
1. Implemented Gumbel-softmax sampler
2. Calibrated classifier
2. Added PyTorch implementation of VAE model and 100K generated samples
2. Connected bits and pieces for seamless workflow 
2. Integrated wandb logger to visualize system utilization, log metrics & artifacts, to track model's performance

### Requirements
To install requirements:
> pip install -r requirements.txt

#### Data
* CIFAR10: the dataset is loaded from the Pytorch 

### Other References:
> https://github.com/pclucas14/pixel-cnn-pp

> https://jyopari.github.io/VAE.html
---
tools & frameworks:
* Pytorch, Tensorflow, wandb
* I have used Google colab for hardware acceleration; dynamic GPU allocation—P100, V100 with 52GB VM system memory mainly for computations.

> assumption: This work is applicable to any SOTA DGM.
