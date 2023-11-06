# -*- coding: utf-8 -*-
import numpy as np
import torch
from features import get_features
from input_output import FileStruct
import input_output as io

from features import get_features
from input_output import FileStruct
import input_output as io
import librosa


# frame-wise embeddings
def embed(embedding_net, features, n_embedding, embed_hop_length, batch_size, device, config):
    with torch.no_grad():
        embedding_net.to(device)
        embedding_net.eval()


        if config.architecture == 'Link_predictor_patches':
            frames = features[0].to(device)
            frames_synced = features[1].to(device)


        else:
            frames = features.to(device)
        

        len_out_embedding = 128  #TODO: This should ultimately be a parameter
        embeddings = torch.empty((len(frames), len_out_embedding))
        idx = 0
        while (idx * batch_size) < len(frames):
            
            if config.architecture in ['Link_predictor_patches', 'ES_GNN_MFCC']:
                batch = frames[idx * batch_size : (idx + 1) * batch_size]
                batch_synced = frames_synced[idx * batch_size : (idx + 1) * batch_size]
                embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch, batch_synced)[-1]
                idx += 1
            else:
                batch = frames[idx * batch_size : (idx + 1) * batch_size]
                if config.training_strategy in ['triplet_cov', 'csn_cov', 'vicreg'] or config.architecture in ['CRNN_4']:
                    embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[0]
                elif config.architecture == 'EmbedNet_CRNN_branches':
                    embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[1]
                elif config.architecture in ['APC', 'Hubert', 'Link_predictor_RNN', 'Link_predictor', 'AN_GCN', 'MLP_', "Link_predictor_2",'Link_predictor_RNN']:
                    embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[-1]
                elif config.architecture in ['AN_GCN_multi_level','Link_predictor_AHC']:
                    if config.annot_level_fine_tune_csn == 0:
                        embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[-1]
                    else:
                        embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[2]
                elif config.architecture in ['ES_GNN', 'MPN']:
                        embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[0]
                elif config.training_strategy in ['csn', 'triplet_off', 'permut', 'supervised','triplet_features']:
                        embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)
                else:
                    embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)[1]
                idx += 1
        del frames
        return embeddings.detach().numpy()
        


def embed_tracks(tracklist, embedding_net, config, device):
    embeddings_list = []
    for track in tracklist:
        embedding = eval_track(track, embedding_net, config, device, config.feat_id)
        embeddings_list.append([track,embedding])
    return embeddings_list



def eval_track(audio_file, embedding_net, config, device, feat_id='mel', feat_type='beat_sync'):
    feat_id = config.feat_id
    features = get_features(audio_file, feat_id, feat_config=config, feat_type=feat_type)
    
    features = (features-np.min(features))/(max(np.max(features)-np.min(features), 1e-6))

    
    features_padded = np.pad(features,
                pad_width=((0, 0),
                            (int(config.embedding.n_embedding//2),
                            int(config.embedding.n_embedding//2))
                            ),
                mode='edge')
    
    
    beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)
    
    features = np.stack([features_padded[:, i:i+config.embedding.n_embedding] for i in beat_frames], axis=0)
    features = torch.tensor(features[:,None, :,:])
            
            

    embeddings = embed(embedding_net, features, config.embedding.n_embedding,
                    config.embedding.embed_hop_length, features.shape[0],
                    device, config) 
        
        
    return embeddings


