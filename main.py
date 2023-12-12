import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

import random

from load_data import load_CelebAHQ256
from util import training_loss, sampling
from util import rescale, find_max_epoch, print_size, model_path, get_dataset_path, results_path

from UNet import UNet


def train(output_directory, ckpt_epoch, n_epochs, learning_rate, batch_size, 
          T, beta_0, beta_T, unet_config, device, data_path, save_path, index):
    """
    Train the UNet model on the CELEBA-HQ 256 * 256 dataset

    Parameters:

    output_directory (str):     save model checkpoints to this path
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded; 
                                automitically selects the maximum epoch if 'max' is selected
    n_epochs (int):             number of epochs to train
    learning_rate (float):      learning rate
    batch_size (int):           batch size
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """
    # Set random seed for reproducibility
    manualSeed = index
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    print(f"Initializing seed {manualSeed}")

    output_directory = save_path

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)


    # Compute diffusion hyperparameters
    Beta = torch.linspace(beta_0, beta_T, T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta + 0
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    # Load training data
    trainloader = load_CelebAHQ256(path=data_path, batch_size=batch_size[device.type])
    print('Data loaded')

    # Predefine model
    net = UNet(device=device, **unet_config).to(device)
    print_size(net)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load checkpoint
    time0 = time.time()
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(output_directory, 'unet_ckpt')
    if ckpt_epoch >= 0:
        model_path = os.path.join(output_directory, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        time0 -= checkpoint['training_time_seconds']
        print('checkpoint model loaded successfully')
    else:
        ckpt_epoch = -1
        print('No valid checkpoint model found, start training from initialization.')

    # Start training
    for epoch in range(ckpt_epoch + 1, n_epochs):
        for i, (X, _) in enumerate(trainloader):
            X = X.to(device)
            
            # Back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), T, X, Alpha_bar, device)
            loss.backward()
            optimizer.step()
            
            # Print training loss
            print(f"epoch: {epoch}/{n_epochs} ({epoch/n_epochs:.3%}%), iter: {i}/{len(trainloader)} ({i/len(trainloader):.3%}%), loss: {loss:.7f}", flush=True)

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_time_seconds': int(time.time()-time0)}, 
                        os.path.join(output_directory, 'unet_ckpt_' + str(epoch) + '.pkl'))
            print('model at epoch %s is saved' % epoch)


def generate(output_directory, ckpt_path, ckpt_epoch, n,
             T, beta_0, beta_T, unet_config, device, results_path, save_path, index):
    """
    Generate images using the pretrained UNet model

    Parameters:

    output_directory (str):     output generated images to this path
    ckpt_path (str):            path of the checkpoints
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded; 
                                automitically selects the maximum epoch if 'max' is selected
    n (int):                    number of images to generate
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """
    # Set random seed for reproducibility
    manualSeed = index
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    print(f"Initializing seed {manualSeed}")

    output_directory = save_path
    ckpt_path = save_path

    # Compute diffusion hyperparameters
    Beta = torch.linspace(beta_0, beta_T, T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta + 0
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    # Predefine model
    net = UNet(**unet_config, device=device).to(device)
    print_size(net)

    # Load checkpoint
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path, 'unet_ckpt')


    model_path = os.path.join(ckpt_path, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
    print(model_path)
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net = UNet(**unet_config, device=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
    except:
        raise Exception('No valid model found')

    # Generation
    time0 = time.time()
    X_gen = sampling(net, (n,3,256,256), T, Alpha, Alpha_bar, Sigma, device)
    print('generated %s samples at epoch %s in %s seconds' % (n, ckpt_epoch, int(time.time()-time0)))

    # Save generated images
    for i in range(n):
        save_image(rescale(X_gen[i]), os.path.join(results_path, 'img_{}.jpg'.format(i)))
    print('saved generated samples at epoch %s' % ckpt_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-t', '--task', type=str, choices=['train', 'generate'],
                        help='Run either training or generation')
    parser.add_argument('--GPU', type=int,
                        help='How many gpus to use')

    parser.add_argument("-M", help="Model name, either GAN or VAE", choices=['diff'])
    parser.add_argument("-D", help="Dataset to be used, either CelebA, Manga109, or Div2k",
                        choices=['CelebA', 'Manga109', 'Div2k'])
    parser.add_argument("-B", help="Batch of model training, used for comparing models in same training batch")
    parser.add_argument("-N", help="Which number of the specific model/dataset is training", type=int)
    parser.add_argument("-A", help="Which type of attack, or no attack, to use",
                        choices=['clean', 'L2_simple_attack'])
    parser.add_argument("--Def", help="Which kind of defense to implement",
                        choices=['noDef', 'pruning', 'activation', 'detection'])

    parser.add_argument("-S", help="Size to use for training", type=int)

    args = parser.parse_args()

    # parse configs
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    unet_config         = config["unet_config"]
    diffusion_config    = config["diffusion_config"]
    train_config        = config["train_config"]
    gen_config          = config["gen_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ngpu = args.GPU
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    data_path = get_dataset_path(args.D, args.A, args.S)

    batch_path = os.path.join("..", "models", args.B)
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)

    save_path = model_path(batch_path, args.M, args.D, args.A, args.Def, args.N)

    results_base = os.path.join("..", "results", args.B)
    if not os.path.exists(results_base):
        os.makedirs(results_base)
    results_path = results_path(results_base, args.M, args.D, args.A, args.Def, args.N)

    # go to task
    if args.task == 'train':
        train(**train_config, **diffusion_config, data_path=data_path, save_path=save_path, index=args.N, unet_config=unet_config, device=device)
    elif args.task =='generate':
        generate(**gen_config, **diffusion_config, results_path=results_path, save_path=save_path, index=args.N, unet_config=unet_config, device=device)
    else:
        raise Exception("Task is not valid.")
    print("Finished")
