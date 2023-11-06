# -*- coding: utf-8 -*-
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

from configuration import config
import input_output as io
from models import EmbedNet
from eval_segmentation import *
import joblib
import warnings
warnings.filterwarnings("ignore")
from utils import clean_tracklist
from embed import embed_tracks


P3s = []
R3s = []
F3s = []


def apply_async_with_callback(embeddings_list, tracklist, config, segmentation, clustering, level, masks, embedding_levels=[], annot=0):

    if segmentation:
        if config.model_name == 'features' and 'SALAMI_2annot' in embeddings_list[0][0]:
            print('eval_segmentation_async_baseline salami')
            #eval_segmentation_async_baseline_salami
            jobs = [ joblib.delayed(eval_segmentation_async_baseline_salami)(audio_file=i[0], config=config, level=level, annot=annot) for i in tqdm(embeddings_list) ]
            out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
            if not config.test:
                out = out[:,1:].astype(float)
            print('P1 =', np.mean(out[:,0]),'+/-', np.std(out[:,0]))
            print('R1 =', np.mean(out[:,1]),'+/-', np.std(out[:,1]))
            print('F1 =', np.mean(out[:,2]),'+/-', np.std(out[:,2]))
            print('P3 =', np.mean(out[:,3]),'+/-', np.std(out[:,3]))
            print('R3 =', np.mean(out[:,4]),'+/-', np.std(out[:,4]))
            print('F3 =', np.mean(out[:,5]),'+/-', np.std(out[:,5]))
            print('PFC =', np.mean(out[:,6]),'+/-', np.std(out[:,6]))
            print('NCE =', np.mean(out[:,7]),'+/-', np.std(out[:,7]))
            return np.mean(out[:,2]), np.mean(out[:,3]), np.mean(out[:,4]), np.mean(out[:,5])

        elif config.model_name == 'features' and 'SALAMI_2annot' not in embeddings_list[0][0]:
            print('eval_segmentation_async_baseline')
            jobs = [ joblib.delayed(eval_segmentation_async_baseline)(audio_file=i[0], config=config, level=level, annot=annot) for i in tqdm(embeddings_list) ]
            out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
            out = np.array(out)
            if not config.test:
                out = out[:,1:].astype(float)
            print('P1 =', np.mean(out[:,0]),'+/-', np.std(out[:,0]))
            print('R1 =', np.mean(out[:,1]),'+/-', np.std(out[:,1]))
            print('F1 =', np.mean(out[:,2]),'+/-', np.std(out[:,2]))
            print('P3 =', np.mean(out[:,3]),'+/-', np.std(out[:,3]))
            print('R3 =', np.mean(out[:,4]),'+/-', np.std(out[:,4]))
            print('F3 =', np.mean(out[:,5]),'+/-', np.std(out[:,5]))
            print('PFC =', np.mean(out[:,6]),'+/-', np.std(out[:,6]))
            print('NCE =', np.mean(out[:,7]),'+/-', np.std(out[:,7]))
            return np.mean(out[:,2]), np.mean(out[:,3]), np.mean(out[:,4]), np.mean(out[:,5])

        elif 'SALAMI_2annot' in embeddings_list[0][0] and (embedding_levels == [] or config.training_strategy not in ['csn', 'csn_supervised']):
            print('eval_segmentation_salami whole')
            jobs = [ joblib.delayed(eval_segmentation_salami_whole)(audio_file=i[0], embeddings=i[1], config=config, level=level) for i in tqdm(embeddings_list) ]
            out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
            out = np.array(out)
            if not config.test:
                out = out[:,1:].astype(float)
            print('Results on SALAMI, annotation level =', level, ', whole embedding matrix')
            print('P1 =', np.mean(out[:,0]),'+/-', np.std(out[:,0]))
            print('R1 =', np.mean(out[:,1]),'+/-', np.std(out[:,1]))
            print('F1 =', np.mean(out[:,2]),'+/-', np.std(out[:,2]))
            print('P3 =', np.mean(out[:,3]),'+/-', np.std(out[:,3]))
            print('R3 =', np.mean(out[:,4]),'+/-', np.std(out[:,4]))
            print('F3 =', np.mean(out[:,5]),'+/-', np.std(out[:,5]))
            print('PFC =', np.mean(out[:,6]),'+/-', np.std(out[:,6]))
            print('NCE =', np.mean(out[:,7]),'+/-', np.std(out[:,7]))
            return np.mean(out[:,2]), np.mean(out[:,3]), np.mean(out[:,4]), np.mean(out[:,5])

            
        else:
            print('eval_segmentation_async')
            jobs = [ joblib.delayed(eval_segmentation_async)(audio_file=i[0], embeddings=i[1], config=config, level=level) for i in tqdm(embeddings_list) ]
            out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
            if config.test:
                out = np.array(out)    
            else:
                out = np.array(out)
                out = out[:,1:].astype(float)
            if len(out.shape)<2:
                print(out)
            print('Results on annotation level =', level, ', whole embedding matrix')
            print('P1 =', np.nanmean(out[:,0]),'+/-', np.nanstd(out[:,0]))
            print('R1 =', np.nanmean(out[:,1]),'+/-', np.nanstd(out[:,1]))
            print('F1 =', np.nanmean(out[:,2]),'+/-', np.nanstd(out[:,2]))
            print('P3 =', np.nanmean(out[:,3]),'+/-', np.nanstd(out[:,3]))
            print('R3 =', np.nanmean(out[:,4]),'+/-', np.nanstd(out[:,4]))
            print('F3 =', np.nanmean(out[:,5]),'+/-', np.nanstd(out[:,5]))
            print('PFC =', np.nanmean(out[:,6]),'+/-', np.nanstd(out[:,6]))
            print('NCE =', np.nanmean(out[:,7]),'+/-', np.nanstd(out[:,7]))
            return out



def main(config):
    feature_dir = os.path.join(config.ds_path, 'features')
    embedding_dir = os.path.join(feature_dir, 'Embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    # loading cuda and model 
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    if config.model_name != 'features':
        
        embedding_net = EmbedNet(config)
        model_path = os.path.join(config.models_dir, config.model_name + "." + 'pt')
        embedding_net.load_state_dict(torch.load(model_path, map_location=device)['state_dict']) 
        embedding_net.to(device)
        embedding_net.eval()
    
        
    print('model loaded')
    # loading and cleaning tracklist
    tracklist = clean_tracklist(config, annotations=True)
    print('tracklist cleaned', len(tracklist))

    if config.model_name != 'features':
        embeddings_list = embed_tracks(tracklist, embedding_net, config, device)
    else:
        embeddings_list = [(i,0) for i in tracklist]
    mask = 0

    #embeddings_list = embed_tracklist(embedding_net, tracklist, device, config)
    print('Device =', device)
    device = 'cpu'
    print('Device =', device)
    print('Length embeddings list =', len(embeddings_list))
    return embeddings_list, tracklist, mask




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('ds_path', type=str, help='Path to the  dataset.')
    parser.add_argument('feat_id', type=str, help='Type of feature before embedding.')
    parser.add_argument('model_name', type=str, help='Model Name')
    parser.add_argument('n_embedding', type=int, help='n_embedding')
    parser.add_argument('level', type=int, help='level')
    parser.add_argument('embed_levels', type=int, help='embed_levels')
    parser.add_argument('mode', type=str, help='mode')
    args, _ = parser.parse_known_args()
    warnings.filterwarnings("ignore")
    config.model_name = args.model_name
    config.ds_path = args.ds_path
    if args.mode == 'seg':
        segmentation = True
        clustering = False
    else:
        segmentation = False
        clustering = True
    config.training_strategy = 'triplet_features'
    config.architecture = 'EmbedNet'
    config.use_batch_norm = True
    config.use_dropout = False
    config.no_cuda = False
    config.feat_id = args.feat_id
    config.nb_workers = 0
    config.feat_type = 'beat_sync'
    config.min_samples = 1
    config.batch_size = 128
    config.embedding.n_embedding = args.n_embedding
    embeddings_list, tracklist, mask = main(config)
    print('Length embeddings list =', len(embeddings_list))
    level = int(args.level)
    e_levels = args.embed_levels
    annot = 0
    config.test = False
    print(config)
    if e_levels == 0:
        embedding_levels = [0]
    elif e_levels == 1:
        embedding_levels = [1]
    else:
        embedding_levels = [0,1]
    print('Embedding levels =',embedding_levels)
    print('Annotation level =', level)
    print('Segmentation =', segmentation)
    print('Clustering =', clustering)
    apply_async_with_callback(embeddings_list, tracklist, config, segmentation, clustering, level, mask, embedding_levels, annot)

        




        
