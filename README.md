PyTorch implementation of [denoising diffusion probabilistic models](https://hojonathanho.github.io/diffusion) on the celebahq (256 * 256) dataset.

# Training:
First, download the celebahq 256 * 256 dataset in [this link](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P). Rename the folder to 'CELEBAHQ' and save it under ```data/```. Then, the images paths are going to be ```data/CELEBAHQ/all/00001.img ~ 30000.img```

## Train from scratch:
Run ```python main.py -c config.json -t 'train'```

## Train from a pretrained model:
Note: training is very slow and batchsize can only be 2 for a single 1080Ti GPU! [This pretrained model](https://drive.google.com/file/d/152nfsDLkpstaIoW7dg4-q3Y4pYAuAn7A/view?usp=sharing) has been trained on 8 1080Ti's for 990 epochs (~450 hours). To use this checkpoint, download it (~1.59GB) and put it under ```model/celebahq/```. Then, run ```python main.py -c config.json -t 'train'```

# Generation:
Download the pretrained model if you don't have one. Then run ```python main.py -c config.json -t 'generate'```. Four images are generated by default. This number can be adjusted by changing ```gen_config["n"]``` in ```config.json```.

# Dependencies:
Necessary packages include Torch, Torchvision, and NumPy. A GPU is needed. If your GPU memory is less than 11GB, then you might need to decrease batchsize ```train_config["batch_size"]``` to 1 in training and decrease the number of images ```gen_config["n"]``` in generation. 
