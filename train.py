# -*- coding: utf-8 -*-
import sys
import os


sys.settrace
import numpy as np
import pandas as pd
import tqdm
import torch
import utils
import torch.optim as optim
from torch.utils.data import DataLoader
from configuration import config as experiment_config
from triplet_features_dset import Triplet_features_dset
from models import EmbedNet
from parser import get_parser





def train_epoch(model, loss_func, device, train_loader, optimizer, disable=False):
    model.train()
    train_losses = []

    for batch_idx, triplets in enumerate(tqdm.tqdm(train_loader,disable=disable)):
        optimizer.zero_grad()
        anchor, positive, negative = triplets
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        embeddings_anchor_norm = model(anchor)
        embeddings_positive_norm = model(positive)
        embeddings_negative_norm = model(negative)
        loss_triplet = loss_func(embeddings_anchor_norm, embeddings_positive_norm, embeddings_negative_norm)
        loss = loss_triplet 
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
            
    return np.mean(train_losses)

    

def train_model(exp_config):
    # Configuration
    
    print(exp_config)
    
    print('NB GPUS = ',torch.cuda.device_count(), 'NB cpus =', os.cpu_count())
    torch.backends.cudnn.benchmark = True
    
    use_cuda = torch.cuda.is_available() and not exp_config.no_cuda
    print('Using GPU:', use_cuda)
    # Create 'models' folder if it does not exist
    exp_config.checkpoints_dir.mkdir(exist_ok=True, parents=True)
    exp_config.models_dir.mkdir(exist_ok=True, parents=True)

    # Seed for reproductible experiments
    torch.manual_seed(exp_config.seed)
    torch.cuda.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Dataset
    train_data = Triplet_features_dset(exp_config)


    dataloader_kwargs = {'num_workers':exp_config.nb_workers,
                        'pin_memory': True} if use_cuda else {}
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Loss function
    loss_func = torch.nn.TripletMarginLoss(margin=exp_config.margin, p=2)

    # Loader
    train_loader = DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=False, **dataloader_kwargs)


    # Network
    embedding_net = EmbedNet(exp_config).to(device)
    model_path = exp_config.models_dir.joinpath(exp_config.model_name+'.pt')

    # Load existing model
    if exp_config.resume:
        embedding_net.load_state_dict(torch.load(model_path)['state_dict']) 
        embedding_net.to(device)
        print('Model loaded')



    # Optimizer
    parameters = filter(lambda p: p.requires_grad, embedding_net.parameters()) 
    optimizer = optim.SGD(parameters, lr=exp_config.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    t = tqdm.trange(1, exp_config.epochs + 1,
                        disable=exp_config.quiet,
                        file=sys.stdout)
    
    # Training
    for epoch in t:
        t.set_description('Training Epoch')
        train_loss = train_epoch(embedding_net, loss_func, device,
                    train_loader, optimizer, disable=exp_config.quiet)        

        print("Epoch {} Training Loss = {}".format(epoch, train_loss))

        utils.save_model(exp_config.checkpoints_dir, exp_config, epoch,
                    embedding_net, optimizer)
        torch.save({'epoch': epoch,
                'state_dict': embedding_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_config': str(exp_config)
                },
            exp_config.models_dir.joinpath(exp_config.model_name + '.pt'))
          

def update_config_with_args(args):
    variables = vars(args)
    for var in variables:
        if variables[var] is not None:
            setattr(experiment_config, var, variables[var])


if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args()
    update_config_with_args(args)