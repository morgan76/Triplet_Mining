# -*- coding: utf-8 -*-
import argparse
from configuration import config
from train import train_model
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exp_unsupervised')
    parser.add_argument('feat_id', type=str,
                        help='Feature used')
    parser.add_argument('ds_path', type=str,
                        help='Path to training dataset')
    args, _ = parser.parse_known_args()

    # Modify things here
    config.model_name = 'test'
    config.ds_path = args.ds_path
    config.training_strategy = 'triplet_features'
    config.feat_type = 'beat_sync'
    config.feat_id = args.feat_id
    config.no_cuda = False
    config.architecture = 'EmbedNet'
    config.seg_algo = 'scluster'
    config.batch_size = 32
    config.resume = False
    config.learning_rate = 1e-3
    config.margin = 0.1
    config.epochs = 1000
    config.nb_workers = 8
    config.feat_type_val = 'beat_sync'
    config.embedding.n_embedding = 512
    warnings.simplefilter('ignore')
    train_model(config)
