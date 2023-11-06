from itertools import filterfalse
import numpy as np
import jams
import mir_eval
import librosa
import os
import torch
import torch.nn.functional as F

from input_output import FileStruct
from utils import times_to_intervals, intervals_to_times, remove_empty_segments
import input_output as io
from algorithms.scluster.main2 import do_segmentation as scluster
from embed import eval_track as embed
from features import get_features
from utils import get_ref_labels
import warnings
warnings.filterwarnings("ignore")

ANNOTATOR = 0
LEVEL = 0



# clean references
def clean(inter, labels):
    i = 0
    labels = list(labels)
    while i<len(inter):
        if np.abs(inter[i,0] - inter[i,1])<0.5:
            inter = np.delete(inter, i, axis=0)
            labels.pop(i)
            i += 1
        else:
            i+= 1   
    inter[0][0] = 0
    return inter, labels

def eval_segmentation_async(audio_file, embeddings, config, level):
    
    # loading annotations
    file_struct = FileStruct(audio_file)
    ref_labels, ref_times, duration = get_ref_labels(file_struct, level, config)

    beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)
    beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
    ref_times[-1] = beat_times[-1]
    
    duration = beat_times[-1]
    
    ref_inter = times_to_intervals(ref_times)
    try:
        ref_inter, ref_labels = clean(ref_inter, ref_labels)
    except:
        print(audio_file)
    
    (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
    
    temp_P1, temp_R1, temp_F1 = [], [], []
    temp_P3, temp_R3, temp_F3 = [], [], []
    temp_PFC, temp_NCE = [], []
    
    est_inter_list, est_labels_list, Cnorm = scluster(embeddings.T, embeddings.T, True)


    for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
        est_idxs = [beat_times[int(i)] for i in est_idxs]
        est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
        est_inter = times_to_intervals(est_idxs)
        est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
        
        P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                est_inter,
                                                window=.5,
                                                trim=True)                                              
        P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                est_inter,
                                                window=2,
                                                trim=True)
        P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                est_inter,
                                                window=3,
                                                trim=True) 

        precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
        S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
        temp_P1.append(P1)
        temp_R1.append(R1)
        temp_F1.append(F1)
        temp_P3.append(P3)
        temp_R3.append(R3)
        temp_F3.append(F3)
        temp_PFC.append(f_PFC)
        temp_NCE.append(S_F)


    ind_max_F1 = np.argmax(temp_F1)
    F1 = temp_F1[ind_max_F1]
    R1 = temp_R1[ind_max_F1]
    P1 = temp_P1[ind_max_F1]
    ind_max_F3 = np.argmax(temp_F3)
    F3 = temp_F3[ind_max_F3]
    R3 = temp_R3[ind_max_F3]
    P3 = temp_P3[ind_max_F3]
    ind_max_PFC = np.argmax(temp_PFC)
    ind_max_NCE = np.argmax(temp_NCE)
    PFC = temp_PFC[ind_max_PFC]
    NCE = temp_NCE[ind_max_NCE]

    if config.test:
        return  P1, R1, F1, P3, R3, F3, PFC, NCE
    else:
        return  audio_file, P1, R1, F1, P3, R3, F3, PFC, NCE
    


def eval_segmentation_salami_whole(audio_file, embeddings, config, level):
    ref_file = FileStruct(audio_file).ref_file
    #try:
    if os.path.isfile(str(ref_file)):
        # loading annotations
        file_struct = FileStruct(audio_file)
        beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
        beat_frames = librosa.util.fix_frames(beat_frames)
        beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
        
        assert np.isinf(embeddings).any() == False
        assert np.isnan(embeddings).any() == False
        #try:
        #    for i in range(len(embeddings)):
        #        embeddings[i] = embeddings[i]/np.linalg.norm(embeddings[i])
        #except:
        #    pass
        embeddings_tensor = torch.tensor(embeddings)
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)
        embeddings_ = embeddings_tensor.numpy()
        est_inter_list, est_labels_list, Cnorm = scluster(embeddings_.T, embeddings_.T, True)

        #est_idxs = librosa.segment.agglomerative(embeddings.T, 5)
        #est_idxs, est_labels, ssm, novelty_curve = foote(embeddings.T, 8, 8, 2, 2, M_gaussian=20, m_median=2, L_peaks=10, bound_norm_feats="min_max")
        #est_idxs, est_labels = cnmf(embeddings)

        temp_P3, temp_R3, temp_F3 = [], [], []
        temp_P1, temp_R1, temp_F1 = [], [], []
        temp_PFC, temp_NCE = [], []

        for annotator in [0, 1]:
            ref_labels, ref_times, duration = get_ref_labels(file_struct, level, config, annotator)
            duration = beat_times[-1]
            ref_times[-1] = beat_times[-1]
            ref_inter = times_to_intervals(ref_times)
            try:
                ref_inter, ref_labels = clean(ref_inter, ref_labels)
            except:
                print(audio_file)
        
            (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
            
            for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                est_idxs = [beat_times[int(i)] for i in est_idxs]
                est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                est_inter = times_to_intervals(est_idxs)
                est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)

                
                P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=.5,
                                                        trim=True)                                              
                P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=2,
                                                        trim=True)
                P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=3,
                                                        trim=True) 
                
                precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                temp_P1.append(P1)
                temp_R1.append(R1)
                temp_F1.append(F1)
                temp_P3.append(P3)
                temp_R3.append(R3)
                temp_F3.append(F3)
                temp_PFC.append(f_PFC)
                temp_NCE.append(S_F)
        
        ind_max_F1 = np.argmax(temp_F1)
        F1 = temp_F1[ind_max_F1]
        R1 = temp_R1[ind_max_F1]
        P1 = temp_P1[ind_max_F1]
        ind_max_F3 = np.argmax(temp_F3)
        F3 = temp_F3[ind_max_F3]
        R3 = temp_R3[ind_max_F3]
        P3 = temp_P3[ind_max_F3]
        ind_max_PFC = np.argmax(temp_PFC)
        ind_max_NCE = np.argmax(temp_NCE)
        PFC = temp_PFC[ind_max_PFC]
        NCE = temp_NCE[ind_max_NCE]
        if config.test:
            return P1, R1, F1, P3, R3, F3, PFC, NCE
        else:
            return  audio_file, P1, R1, F1, P3, R3, F3, PFC, NCE
    #except:
    #    return  0,0,0, 0,0,0, 0, 0





def eval_segmentation_salami_levels(audio_file, embeddings, config, embedding_levels, mask):
    ref_file = FileStruct(audio_file).ref_file
    #try:
    if os.path.isfile(str(ref_file)):
        # loading annotations
        file_struct = FileStruct(audio_file)
        beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
        beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)

        #est_idxs = librosa.segment.agglomerative(embeddings.T, 5)
        #est_idxs, est_labels, ssm, novelty_curve = foote(embeddings.T, 8, 8, 2, 2, M_gaussian=20, m_median=2, L_peaks=10, bound_norm_feats="min_max")
        #est_idxs, est_labels = cnmf(embeddings)

        temp_P3, temp_R3, temp_F3 = [], [], []
        temp_P1, temp_R1, temp_F1 = [], [], []
        temp_PFC, temp_NCE = [], []

        for embedding_level in embedding_levels:
            embeddings_ = embeddings*mask[embedding_level]
            embeddings_tensor = torch.tensor(embeddings_)
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)
            embeddings_ = embeddings_tensor.numpy()
            est_inter_list, est_labels_list, Cnorm = scluster(embeddings_.T, embeddings_.T, True)

            for annotator in [0,1]:
                ref_labels, ref_times, duration = get_ref_labels(file_struct, 0, config, annotator)
                ref_inter = times_to_intervals(ref_times)
                try:
                    ref_inter, ref_labels = clean(ref_inter, ref_labels)
                except:
                    print(audio_file)
            
                (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
                
                for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                    est_idxs = [beat_times[int(i)] for i in est_idxs]
                    est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                    est_inter = times_to_intervals(est_idxs)
                    est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)

                    
                    P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=.5,
                                                            trim=True)                                              
                    P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=2,
                                                            trim=True)
                    P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=3,
                                                            trim=True) 
                    
                    precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                    S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                    temp_P1.append(P1)
                    temp_R1.append(R1)
                    temp_F1.append(F1)
                    temp_P3.append(P3)
                    temp_R3.append(R3)
                    temp_F3.append(F3)
                    temp_PFC.append(f_PFC)
                    temp_NCE.append(S_F)
        

        ind_max_F1 = np.argmax(temp_F1)
        F1_up = temp_F1[ind_max_F1]
        R1_up = temp_R1[ind_max_F1]
        P1_up = temp_P1[ind_max_F1]
        ind_max_F3 = np.argmax(temp_F3)
        F3_up = temp_F3[ind_max_F3]
        R3_up = temp_R3[ind_max_F3]
        P3_up = temp_P3[ind_max_F3]
        ind_max_PFC = np.argmax(temp_PFC)
        ind_max_NCE = np.argmax(temp_NCE)
        PFC_up = temp_PFC[ind_max_PFC]
        NCE_up = temp_NCE[ind_max_NCE]



        temp_P3, temp_R3, temp_F3 = [], [], []
        temp_P1, temp_R1, temp_F1 = [], [], []
        temp_PFC, temp_NCE = [], []


        for embedding_level in embedding_levels:
            embeddings_ = embeddings*mask[embedding_level]
            embeddings_tensor = torch.tensor(embeddings_)
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)
            embeddings_ = embeddings_tensor.numpy()
            est_inter_list, est_labels_list, Cnorm = scluster(embeddings_.T, embeddings_.T, True)

            for annotator in [0,1]:
                ref_labels, ref_times, duration = get_ref_labels(file_struct, 1, config, annotator)
                ref_inter = times_to_intervals(ref_times)
                try:
                    ref_inter, ref_labels = clean(ref_inter, ref_labels)
                except:
                    print(audio_file)
            
                (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
                
                for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                    est_idxs = [beat_times[int(i)] for i in est_idxs]
                    est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                    est_inter = times_to_intervals(est_idxs)
                    est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)

                    
                    P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=.5,
                                                            trim=True)                                              
                    P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=2,
                                                            trim=True)
                    P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=3,
                                                            trim=True) 
                    
                    precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                    S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                    temp_P1.append(P1)
                    temp_R1.append(R1)
                    temp_F1.append(F1)
                    temp_P3.append(P3)
                    temp_R3.append(R3)
                    temp_F3.append(F3)
                    temp_PFC.append(f_PFC)
                    temp_NCE.append(S_F)
        

        ind_max_F1 = np.argmax(temp_F1)
        F1_low = temp_F1[ind_max_F1]
        R1_low = temp_R1[ind_max_F1]
        P1_low = temp_P1[ind_max_F1]
        ind_max_F3 = np.argmax(temp_F3)
        F3_low = temp_F3[ind_max_F3]
        R3_low = temp_R3[ind_max_F3]
        P3_low = temp_P3[ind_max_F3]
        ind_max_PFC = np.argmax(temp_PFC)
        ind_max_NCE = np.argmax(temp_NCE)
        PFC_low = temp_PFC[ind_max_PFC]
        NCE_low = temp_NCE[ind_max_NCE]

        if config.test:
            return  P1_up, R1_up, F1_up, P3_up, R3_up, F3_up, PFC_up, NCE_up, P1_low, R1_low, F1_low, P3_low, R3_low, F3_low, PFC_low, NCE_low
        else:
            return  audio_file, P1_up, R1_up, F1_up, P3_up, R3_up, F3_up, PFC_up, NCE_up, P1_low, R1_low, F1_low, P3_low, R3_low, F3_low, PFC_low, NCE_low
    #except:
    #    return  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0




def eval_segmentation_async_baseline(audio_file, config, level, annot=0):
    ref_file = FileStruct(audio_file).ref_file
    #print('IN EVAL SEGMENTATION', flush=True)
    try:
        if os.path.isfile(str(ref_file)):
            # loading annotations
            file_struct = FileStruct(audio_file)
            ref_labels, ref_times, duration = get_ref_labels(file_struct, level, config)
            beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
            beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
            ref_inter = times_to_intervals(ref_times)
            try:
                ref_inter, ref_labels = clean(ref_inter, ref_labels)
            except:
                print(audio_file)
            
            (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)

            
            mfcc = get_features(audio_file, 'mfcc', feat_config=config, feat_type=config.feat_type)
            cqt = get_features(audio_file, 'cqt', feat_config=config, feat_type=config.feat_type)
            beat_frames = librosa.util.fix_frames(beat_frames)
            beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
            mfcc = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)
            cqt = librosa.util.sync(cqt, beat_frames, aggregate=np.mean)
            est_inter_list, est_labels_list, Cnorm = scluster(cqt, mfcc, True)
            
            temp_P3, temp_R3, temp_F3 = [], [], []
            temp_P1, temp_R1, temp_F1 = [], [], []
            temp_PFC, temp_NCE = [], []
            for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                est_idxs = [beat_times[int(i)] for i in est_idxs]
                est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                est_inter = times_to_intervals(est_idxs)
                est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
                #est_idxs = intervals_to_times(est_inter)
                #est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                #est_inter = times_to_intervals(est_idxs)
                
                P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=.5,
                                                        trim=True)                                              
                P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=2,
                                                        trim=True)
                P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                        est_inter,
                                                        window=3,
                                                        trim=True) 
                #ref_inter, ref_labels = mir_eval.util.adjust_intervals(ref_inter, ref_labels, t_min=0, t_max=duration)
                #est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
                precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                temp_P1.append(P1)
                temp_R1.append(R1)
                temp_F1.append(F1)
                temp_P3.append(P3)
                temp_R3.append(R3)
                temp_F3.append(F3)
                temp_PFC.append(f_PFC)
                temp_NCE.append(S_F)
            

            ind_max_F1 = np.argmax(temp_F1)
            F1 = temp_F1[ind_max_F1]
            R1 = temp_R1[ind_max_F1]
            P1 = temp_P1[ind_max_F1]
            ind_max_F3 = np.argmax(temp_F3)
            F3 = temp_F3[ind_max_F3]
            R3 = temp_R3[ind_max_F3]
            P3 = temp_P3[ind_max_F3]
            ind_max_PFC = np.argmax(temp_PFC)
            ind_max_NCE = np.argmax(temp_NCE)
            PFC = temp_PFC[ind_max_PFC]
            NCE = temp_NCE[ind_max_NCE]
            if config.test:
                return  P1, R1, F1, P3, R3, F3, PFC, NCE
            else:
                return  audio_file, P1, R1, F1, P3, R3, F3, PFC, NCE
    except:
        return  0,0,0, 0,0,0, 0, 0



def eval_segmentation_async_baseline_salami(audio_file, config, level, annot=0):
    ref_file = FileStruct(audio_file).ref_file
    #print('IN EVAL SEGMENTATION', flush=True)
    try:
        if os.path.isfile(str(ref_file)):
            # loading annotations
            file_struct = FileStruct(audio_file)
            
            beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
            beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)

            
            mfcc = get_features(audio_file, 'mfcc', feat_config=config, feat_type=config.feat_type)
            cqt = get_features(audio_file, 'cqt', feat_config=config, feat_type=config.feat_type)
            beat_frames = librosa.util.fix_frames(beat_frames)
            beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
            mfcc = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)
            cqt = librosa.util.sync(cqt, beat_frames, aggregate=np.mean)
            est_inter_list, est_labels_list, Cnorm = scluster(cqt, mfcc, True)
            
            temp_P3, temp_R3, temp_F3 = [], [], []
            temp_P1, temp_R1, temp_F1 = [], [], []
            temp_PFC, temp_NCE = [], []

            for annot in [0,1]:
                ref_labels, ref_times, duration = get_ref_labels(file_struct, level, config, annot)
                ref_inter = times_to_intervals(ref_times)
                try:
                    ref_inter, ref_labels = clean(ref_inter, ref_labels)
                except:
                    print(audio_file)
                
                (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)

                for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                    est_idxs = [beat_times[int(i)] for i in est_idxs]
                    est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                    est_inter = times_to_intervals(est_idxs)
                    est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
                    #est_idxs = intervals_to_times(est_inter)
                    #est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                    #est_inter = times_to_intervals(est_idxs)
                    
                    P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=.5,
                                                            trim=True)                                              
                    P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=2,
                                                            trim=True)
                    P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=3,
                                                            trim=True) 
                    #ref_inter, ref_labels = mir_eval.util.adjust_intervals(ref_inter, ref_labels, t_min=0, t_max=duration)
                    #est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
                    precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                    S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                    temp_P1.append(P1)
                    temp_R1.append(R1)
                    temp_F1.append(F1)
                    temp_P3.append(P3)
                    temp_R3.append(R3)
                    temp_F3.append(F3)
                    temp_PFC.append(f_PFC)
                    temp_NCE.append(S_F)
            

            ind_max_F1 = np.argmax(temp_F1)
            F1 = temp_F1[ind_max_F1]
            R1 = temp_R1[ind_max_F1]
            P1 = temp_P1[ind_max_F1]
            ind_max_F3 = np.argmax(temp_F3)
            F3 = temp_F3[ind_max_F3]
            R3 = temp_R3[ind_max_F3]
            P3 = temp_P3[ind_max_F3]
            ind_max_PFC = np.argmax(temp_PFC)
            ind_max_NCE = np.argmax(temp_NCE)
            PFC = temp_PFC[ind_max_PFC]
            NCE = temp_NCE[ind_max_NCE]
            if config.test:
                return P1, R1, F1, P3, R3, F3, PFC, NCE
            else:
                return  audio_file, P1, R1, F1, P3, R3, F3, PFC, NCE
    except:
        return  0,0,0, 0,0,0, 0, 0




def eval_segmentation_async_best(audio_file, embeddings, config, level, mask):
    ref_file = FileStruct(audio_file).ref_file
    try:
        if os.path.isfile(str(ref_file)):
            # loading annotations
            beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
            beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
        
            temp_P1, temp_R1, temp_F1 = [], [], []
            temp_P3, temp_R3, temp_F3 = [], [], []
            temp_PFC, temp_NCE = [], []


            for condition in range(config.n_conditions):
                embeddings_ = embeddings*mask[condition]
                #if config.training_strategy == 'csn':
                    #for i in range(len(embeddings_)):
                    #    embeddings_[i] = embeddings_[i]/np.linalg.norm(embeddings_[i])
                est_inter_list, est_labels_list, Cnorm = scluster(embeddings_.T, embeddings_.T, True)


                for annot in [0,1]:
                    file_struct = FileStruct(audio_file)
                    ref_labels, ref_times, duration = get_ref_labels(file_struct, level, config, annot)
                    ref_inter = times_to_intervals(ref_times)
                    try:
                        ref_inter, ref_labels = clean(ref_inter, ref_labels)
                    except:
                        print(audio_file)
                    
                    (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
               
                
                    for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                        est_idxs = [beat_times[int(i)] for i in est_idxs]
                        est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                        est_inter = times_to_intervals(est_idxs)
                        est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
                        
                        P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                                est_inter,
                                                                window=.5,
                                                                trim=True)                                              
                        P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                                est_inter,
                                                                window=2,
                                                                trim=True)
                        P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                                est_inter,
                                                                window=3,
                                                                trim=True) 

                        precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                        S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                        temp_P1.append(P1)
                        temp_R1.append(R1)
                        temp_F1.append(F1)
                        temp_P3.append(P3)
                        temp_R3.append(R3)
                        temp_F3.append(F3)
                        temp_PFC.append(f_PFC)
                        temp_NCE.append(S_F)
            

            ind_max_F1 = np.argmax(temp_F1)
            F1 = temp_F1[ind_max_F1]
            R1 = temp_R1[ind_max_F1]
            P1 = temp_P1[ind_max_F1]
            ind_max_F3 = np.argmax(temp_F3)
            F3 = temp_F3[ind_max_F3]
            R3 = temp_R3[ind_max_F3]
            P3 = temp_P3[ind_max_F3]
            ind_max_PFC = np.argmax(temp_PFC)
            ind_max_NCE = np.argmax(temp_NCE)
            PFC = temp_PFC[ind_max_PFC]
            NCE = temp_NCE[ind_max_NCE]
            if config.test:
                return  P1, R1, F1, P3, R3, F3, PFC, NCE
            else:
                return  audio_file, P1, R1, F1, P3, R3, F3, PFC, NCE
    except:
        return  0,0,0, 0,0,0, 0, 0


def eval_segmentation(audio_file, embedding_net, config, device, feat_id, return_data=False):
    # REFERENCE
    feat_id = config.feat_id
    ref_file = FileStruct(audio_file).ref_file
    #print('IN EVAL SEGMENTATION')
    if os.path.isfile(str(ref_file)):
        # loading annotations

        file_struct = FileStruct(audio_file)
        beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
        beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
        ref_labels, ref_times, duration = get_ref_labels(file_struct, LEVEL, config)
        try:
            ref_inter, ref_labels = clean(ref_inter, ref_labels)
        except:
            print(audio_file)
        
        (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
        
        # loading beat times
        
        
        # calculating embeddings
        embeddings = embed(audio_file, embedding_net, config, device, feat_id, feat_type='beat_sync')

        if config.feat_type_val == 'beat_sync':

            if config.seg_algo == 'scluster':

                #print('ENTERED IN SCLUSTER')
                temp_F3, temp_R3, temp_P3, temp_PFC, temp_S_F = [], [], [], [], []
                temp_F1, temp_R1, temp_P1 = [], [], []
                est_inter_list, est_labels_list, Cnorm = scluster(embeddings.T, embeddings.T, True)

                for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
                    if config.feat_type_val == 'beat_sync':
                        est_idxs = [beat_times[int(i)] for i in est_idxs]
                    else:
                        est_idxs = librosa.frames_to_time(est_idxs, sr=config.sample_rate, hop_length=config.hop_length*config.embedding.embed_hop_length)
                    
                    est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
                
                    est_inter = times_to_intervals(est_idxs)

                    est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
     
                    P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=.5,
                                                            trim=True)                                              
                    P2, R2, F2 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=2,
                                                            trim=True)
                    P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                            est_inter,
                                                            window=3,
                                                            trim=True) 

                    precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
                    S_over, S_under, S_F = mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
                    
                    temp_S_F.append(S_F)
                    temp_PFC.append(f_PFC)
                    temp_P3.append(P3)
                    temp_R3.append(R3)
                    temp_F3.append(F3)
                    temp_P1.append(P1)
                    temp_R1.append(R1)
                    temp_F1.append(F1)
                    ssm, novelty_curve = 0, 0
                
                max_ind = np.argmax(temp_F3)
                P3 = temp_P3[max_ind]
                R3 = temp_R3[max_ind]
                F3 = temp_F3[max_ind]
                P1 = temp_P1[max_ind]
                R1 = temp_R1[max_ind]
                F1 = temp_F1[max_ind]
                PFC = temp_PFC[max_ind]
                S_F = temp_S_F[max_ind]
                est_inter = est_inter_list[max_ind]
                
                est_labels = est_labels_list
                est_inter = est_inter_list

                ref_labels_up, ref_times_up, duration = get_ref_labels(file_struct, 0, config)
                ref_inter_up = times_to_intervals(ref_times_up)
                ref_labels_low, ref_times_low, duration = get_ref_labels(file_struct, 1, config)
                ref_inter_low = times_to_intervals(ref_times_low)
                
                ref_inter = [ref_inter_up, ref_inter_low]
                ref_labels = [ref_labels_up, ref_labels_low]
        
        results1 = {"P1": P1,
                    "R1": R1,
                    "F1": F1}
        results2 = {"PFC": PFC,
                    "S_F": S_F}
        results3 = {"P3": P3,
                    "R3": R3,
                    "F3": F3}

        if return_data:
            return results1, results2, results3, ssm, novelty_curve, embeddings, ref_inter, est_inter, ref_labels, est_labels, max_ind
        else:
            return results1, results2, results3


