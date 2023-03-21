# how many images to generate from the latent space
import torch
import argparse
import numpy as np

# model skeleton
vae = VAE()
# load last vae checkpoint
final_model = glob.glob("./vae_checkpoints/*.pt")[-1]
vae.load_state_dict(torch.load(final_model))
# disable dropout etc.
vae.eval()

def return_npz_file():
    x,y,z = args.train, args.val, args.test
    train_data = np.savez_compressed(vae.decode(torch.randn(x,180,1,1)))
    valid_data = np.savez_compressed(vae.decode(torch.randn(y,180,1,1)))
    test_data = np.savez_compressed(vae.decode(torch.randn(z,180,1,1)))
    
    np.savez_compressed('./gen/gen_samples.npz', train_data = train_data[0], valid_data = valid_data[0],
                    test_data = test_data[0])

parser = argparse.ArgumentParser(description='generate images through a learned VAE')
parser.add_argument('--train', type=int, default=1000)
parser.add_argument('--val', type=int, default=1000)
parser.add_argument('--test', type=int, default=1000)

args = parser.parse_args()



