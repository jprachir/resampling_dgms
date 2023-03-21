## PyTorch implementation of PixelCNN++
## Reference
Based on this repository, we are trying to generate the data:

A Pytorch Implementation of [PixelCNN++.](https://github.com/pclucas14/pixel-cnn-pp)

## Task & framework details:
We want to generate train data of a size 45k samples and test data of size 100k from the PixelCNN++ model. This large data is required to analyze confidence interval. Currently we are using Google colab for hardware acceleration- which allocates a dynamic GPU —either P100 or V100, memory of 16GB-- with a 52GB VM system memory mainly for computations.

## Limitation of this code: 
It's taking almost one hour to generate 250 samples, since this model generates data pixel by pixel. In this sense to generate 100k samples each with 32x32 dimensions model will take around 15 days on just one gpu. 

## Possible acceleration: 
With my limited knowledge I could suggest you may be increasing the number of workers (current is 8 workers) or increasing the sample batch size (current is 250) according to how much gpu memory handles.

## How to run this code

##### If you want to run this in an isolated python environment, please follow the given steps
- ```pip install virtualenv```
- ```cd to my-project/ ``` (possibly you are in the project directory)
- ```virtualenv venv``` setup virual environment for the project
- ```source venv/bin/activate``` activating venv, you can see venv apearing at the beginning of the terminla prompt
- continue with the following code like installing dependencies and running the scripts etc.
- ```deactivate``` to leave the virtual environment

*generate_sample.py* script handles importing the best model, and generating required number of samples by performing the forward pass 
No. of model parameters: ~53M 

###### Default configuration for the script generate_sample.py 
##### required data I/O arguments
> -i --data_dir ./data
> -d --dataset cifar10
> -g --number_of_samples 45000
##### best model arguments
> -q --nr_resenet 5 
> -n --nr_filters 160
> -l --lr 0.0004

## Datasets
The CIFAR10 dataset will be downloaded automatically in the data folder for the first time when script starts running.

## Pretrained Models and Samples

Best pretrained ckpt for this model is stored in the model_best_chkpoint folder with ".pt" format. One sample image is stored in images folder.

Best model naming convention:
>*model_lr.0.00040_nr-resnet5_nr-filters160_889*
--
>*model_lr_no-of-residual-blocks_no-of-filters-to-use-across-the-model-the-higher-the-larger-model_epoch*

---
## How to run this code
### 1. Requirements
The codebase is implemented in Python 3.7. 

To install the necessary requirements, run the following commands:
```
pip install -r requirements.txt
```
1. Run this cmd in the terminal to just generate training data by default
```
python generate_sample.py 
```
which will automatically starts generating samples in the folder all_train_batches_45k. The generation is in the numpy (.npz) format and in the batches of 250 samples per iteration (180 such iterations, calculated automatically).

2. Run this cmd in the terminal to just generate testing data
Please edit the last line (#134) in generate_sample.py (it's about changing only the directory path)
from    
```
np.savez_compressed("./generated_data/all_train_batches_45k/batch_{}.npz".format(i), batch_250)
```
to
```
np.savez_compressed("./generated_data/all_test_batches_100k/batch_{}.npz".format(i), batch_250)
```

and run following command in the terminal
```
python generate_sample.py -g 100000
```
which will automatically starts generating samples in the folder all_test_batches_100k. The generation is in the numpy (.npz) format and in the batches of 250 samples per iteration (400 such iterations, calculated automatically).

#### Secondary reference:
- [virtualenv](https://sourabhbajaj.com/mac-setup/Python/virtualenv.html)