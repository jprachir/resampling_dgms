from random import randint
from torchvision.utils import save_image
from IPython.display import Image
from IPython.core.display import Image, display
import torch
import torchvision
from torchvision import datasets, transforms
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def compare(x):
    recon_x, _, _ = vae(x)
    return torch.cat([x, recon_x])

def actual_vs_reconstructed_image():
    vae = VAE()
    # load last vae checkpoint
    final_model = glob.glob("./vae_checkpoints/*.pt")[-1]
    vae.load_state_dict(torch.load(final_model))
    # disable dropout etc.
    vae.eval()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


    fixed_x = testset[randint(1, 100)][0].unsqueeze(0)
    compare_x = compare(fixed_x)

    save_image(compare_x.data.cpu(), 'x_&_x_hat.png')
    display(Image('x_&_x_hat.png', width=700, unconfined=True))

def vis_random_gen():
    img_arr = np.load('./gen_data/gen_samples.npz')['train_data']
    img_arr = np.random.choice(img_arr,size=4,replace=False,)
    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111, 
                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes
                 )

    for ax, im in zip(grid, img_arr):
        ax.imshow(im)

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    