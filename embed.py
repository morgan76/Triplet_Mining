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
        frames = features.to(device)
        len_out_embedding = 128 
        embeddings = torch.empty((len(frames), len_out_embedding))
        idx = 0
        while (idx * batch_size) < len(frames):
            batch = frames[idx * batch_size : (idx + 1) * batch_size]
            embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)
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


