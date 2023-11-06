import numpy as np
import torch
import librosa
from features import get_features
import input_output as io
from utils import clean_tracklist
from scipy import ndimage
import joblib
from scipy.ndimage import filters
from scipy.ndimage import median_filter
from tqdm import tqdm




class Triplet_features_dset(torch.utils.data.Dataset):

    def __init__(self, experiment_config, feature_types=['mfcc','chroma'], W=16, ALPHA = 60, BETA = .85, alpha_neg = 5, LAMBDA = .5):

        self.experiment_config = experiment_config
        self.beta_ssm = .99
        self.W = W
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.alpha_neg = alpha_neg
        self.LAMBDA = LAMBDA
        self.feature_types = feature_types
        self.tracklist = clean_tracklist(experiment_config, annotations=False)
        print('Length training tracklist = {}'.format(len(self.tracklist)))
        self.triplets = self.buildTriplets()
        print('Overall number of triplets = {}'.format(len(self.triplets)))


    def gaussian_filter(self, X, M=8, axis=0):
        for i in range(X.shape[axis]):
            if axis == 1:
                X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
            elif axis == 0:
                X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
        return X
    

    def buildTriplets_(self):
        print('Sampling triplets...')
        triplets = []
        tracks = self.tracklist
        n_per_songs = self.experiment_config.batch_size
        for track in tqdm(tracks):
            file_struct = io.FileStruct(track)
            triplets_track = []
            beat_frames = io.read_beats(file_struct.beat_file)
            beat_frames = librosa.util.fix_frames(beat_frames)
            mfcc = get_features(track, self.feature_types[0], self.experiment_config, self.experiment_config.feat_type)
            chromas = get_features(track, self.feature_types[1], self.experiment_config, self.experiment_config.feat_type)
            mfcc_synced = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)[1:]
            mfcc_synced = ndimage.median_filter(mfcc_synced, size=(1, 8))
            chromas_synced = librosa.util.sync(chromas, beat_frames, aggregate=np.mean)
            chromas_synced = ndimage.median_filter(chromas_synced, size=(1, 8))
            SSM, SSM_neg = self.compute_ssms(mfcc_synced, chromas_synced)
            triplets_track = self.get_triplets(track, SSM, SSM_neg, n_per_songs, beat_frames)
        
            assert len(triplets_track) == n_per_songs

            for triplet in triplets_track:
                triplets.append(triplet)

        return triplets
   

    def buildTriplets(self):
        tracks = self.tracklist
        n_per_song = self.experiment_config.batch_size
        print('Sampling triplets...')
        triplets = []
        jobs = [joblib.delayed(self.sample_triplets_track)(track=track, n_per_songs=n_per_song) for track in tracks]
        out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
        for triplets_ in out:
            triplets += triplets_
        return triplets


    def sample_triplets_track(self, track, n_per_songs):
        file_struct = io.FileStruct(track)
        triplets_track = []
        #try:

        # loading beat file
        beat_frames = io.read_beats(file_struct.beat_file)
        beat_frames = librosa.util.fix_frames(beat_frames)

        # loading MFCC and chroma
        mfcc = get_features(track, self.feature_types[0], self.experiment_config, self.experiment_config.feat_type)
        chromas = get_features(track, self.feature_types[1], self.experiment_config, self.experiment_config.feat_type)

        # beat synchronization & median filtering
        mfcc_synced = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)[1:]
        mfcc_synced = ndimage.median_filter(mfcc_synced, size=(1, 8))
        chromas_synced = librosa.util.sync(chromas, beat_frames, aggregate=np.mean)
        chromas_synced = ndimage.median_filter(chromas_synced, size=(1, 8))

        # positive and negative sampling matrices
        SSM, SSM_neg = self.compute_ssms(mfcc_synced, chromas_synced)

        # triplet sampling
        triplets_track = self.get_triplets(track, SSM, SSM_neg, n_per_songs, beat_frames)
        return triplets_track
        #except:
        #    return []


    def get_triplets(self, track, SSM, SSM_neg, n_per_songs, beat_frames):
        triplets_track = []
        try:
            L = len(SSM)
            try:
                anchors_total = np.random.choice(np.arange(len(SSM)), size=n_per_songs, replace=False)
            except:
                anchors_total = np.random.choice(np.arange(len(SSM)), size=n_per_songs, replace=True)
            for anchor in anchors_total:
            
                positive = np.random.choice(np.arange(len(SSM)), size=1, p=SSM[anchor]/np.sum(SSM[anchor]))[0]
            
                negative = np.random.choice(np.arange(len(SSM)), size=1, p=SSM_neg[anchor]/np.sum(SSM_neg[anchor]))[0]

                triplets_track.append([track, (beat_frames[anchor], beat_frames[positive], beat_frames[negative])])

            assert len(triplets_track) == n_per_songs
            return triplets_track
        except:
            return triplets_track


    def sigmoid(self, x, beta, offset):
        return 1/(1+np.exp(-beta*(x-offset)))


    def compute_ssm(self, X, delay):
        if delay != 0:
            X_stack = librosa.feature.stack_memory(X, n_steps=delay, delay=1)
            R_pos = librosa.segment.recurrence_matrix(X_stack, mode='affinity', width=2, k=X.shape[1]).astype(float)#
            for i in range(len(R_pos)):
                R_pos[i] = (R_pos[i]-np.min(R_pos[i]))/(max(np.max(R_pos[i])-np.min(R_pos[i]), 1e-6))
                R_pos[i,i] = 0
            return R_pos
        else:
            R_pos = librosa.segment.recurrence_matrix(X, mode='affinity', width=2, k=X.shape[1]).astype(float)#
            for i in range(len(R_pos)):
                R_pos[i] = (R_pos[i]-np.min(R_pos[i]))/(max(np.max(R_pos[i])-np.min(R_pos[i]), 1e-6))
                R_pos[i,i] = 0
            return R_pos


    def compute_ssms(self, mfcc, chroma):
        # convert features to time-larg and calculate SSM
        SSM_chromas = self.compute_ssm(chroma, 16)
        SSM_mfcc = self.compute_ssm(mfcc, 8)

        # diagonal filtering
        diagonal_median = librosa.segment.timelag_filter(median_filter)
        
        # sigmoid filtering
        for i in range(len(SSM_chromas)):
            SSM_chromas[i] = self.sigmoid(SSM_chromas[i], self.ALPHA, self.BETA)
            SSM_mfcc[i] = self.sigmoid(SSM_mfcc[i], self.ALPHA, self.BETA)
        
        # SSMS combination with balance parameter LAMBDA (GAMMA in the paper)
        SSM = self.LAMBDA*SSM_chromas+(1-self.LAMBDA)*SSM_mfcc
        SSM = diagonal_median(SSM, size=(1, 31), mode='mirror')

        # row-wise normalization & second sigmoid filtering
        for i in range(len(SSM)):
            SSM[i] = (SSM[i]-np.min(SSM[i]))/(max(np.max(SSM[i])-np.min(SSM[i]), 1e-6))
            SSM[i] = self.sigmoid(SSM[i], self.ALPHA, self.beta_ssm)

        # gaussian filter to enlarge repeating regions
        SSM_gaussian = self.gaussian_filter(SSM.T, M=self.W, axis=1)
        SSM_gaussian = self.gaussian_filter(SSM.T, M=self.W, axis=0)


        SSM = self.LAMBDA*SSM_chromas+(1-self.LAMBDA)*SSM_mfcc
        SSM = diagonal_median(SSM, size=(1, 31), mode='mirror')
        for i in range(len(SSM)):
            SSM[i] = (SSM[i]-np.min(SSM[i]))/(max(np.max(SSM[i])-np.min(SSM[i]), 1e-6))
            SSM[i] = self.sigmoid(SSM[i], self.ALPHA, self.beta_ssm)

        # negative sampling matrices
        negatives_gaussian = self.gaussian_filter(1-SSM.T, M=self.W, axis=1)
        negatives_gaussian = self.gaussian_filter(1-SSM.T, M=self.W, axis=0)    

        # row-wise normalization
        for i in range(len(SSM_gaussian)):
            SSM_gaussian[i] = (SSM_gaussian[i]-np.min(SSM_gaussian[i]))/(max(np.max(SSM_gaussian[i])-np.min(SSM_gaussian[i]), 1e-6))

        # exponential decay
        for i in range(len(negatives_gaussian)):
            for j in range(len(negatives_gaussian)):
                num = max(np.abs(i-j)/len(negatives_gaussian), SSM_gaussian[i,j])
                negatives_gaussian[i,j] *= np.exp(-self.alpha_neg*num) 

        return SSM_gaussian, negatives_gaussian


    def __getitem__(self, index):
        track, (anchor_index, positive_index, negative_index) = self.triplets[index]
        features = get_features(track, self.experiment_config.feat_id, self.experiment_config, self.experiment_config.feat_type)
        features = (features-np.min(features))/(max(np.max(features)-np.min(features), 1e-6))
        features_padded = np.pad(features,
                    pad_width=((0, 0),
                                (int(self.experiment_config.embedding.n_embedding//2),
                                int(self.experiment_config.embedding.n_embedding//2))
                                ),
                    mode='edge')
        anchor_patch = torch.as_tensor(features_padded[:, anchor_index:anchor_index+self.experiment_config.embedding.n_embedding][None, :, :])
        positive_patch = torch.as_tensor(features_padded[:, positive_index:positive_index+self.experiment_config.embedding.n_embedding][None, :, :])
        negative_patch = torch.as_tensor(features_padded[:, negative_index:negative_index+self.experiment_config.embedding.n_embedding][None, :, :])
        return anchor_patch, positive_patch, negative_patch


    def __len__(self):
        return len(self.triplets)
