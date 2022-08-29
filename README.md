## Implementation of a given paper
### Primary Reference:
> Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting  
> [Aditya Grover](https://aditya-grover.github.io), Jiaming Song, Alekh Agarwal, Kenneth Tran, Ashish Kapoor, Eric Horvitz, Stefano Ermon.  
> Paper: https://arxiv.org/abs/1906.09531  

### Differences with official implementation
1. Added PyTorch implementation of VAE models and 100K generated samples
2. Connected bits and pieces for seamless workflow 
3. Added results folder

### Other References:
> https://github.com/pclucas14/pixel-cnn-pp
> https://jyopari.github.io/VAE.html
---
I have used Google colab for hardware acceleration; dynamic GPU allocation—P100, V100 with 52GB VM system memory mainly for computations.

> assumption: This work is applicable to any SOTA DGM.

> Remaining tasks:
- Implement SIR algorithm

> Possible improvements:
- Improve the sampler
- Integrate domain knowledge
- Analyse different SOTA DGMs
- Integrate XAI
