# -*- coding: utf-8 -*-
import argparse
import multiprocessing as mp
from pathlib import Path
import os
import numpy as np
import librosa
from tqdm import tqdm
import madmom

from configuration import config
from utils import valid_feat_files
from input_output import FileStruct, write_features, update_beats


def compute_features(y, feat_id, feat_config):
    if feat_id == 'cqt':
        features = librosa.cqt(y=y, sr=feat_config.sample_rate,
                                hop_length=feat_config.hop_length,
                                n_bins=6*12, 
                                bins_per_octave=12)
        features = librosa.power_to_db(np.abs(features)**2)
    if feat_id == 'mfcc':
        features = librosa.feature.mfcc(y=y, sr=feat_config.sample_rate,
                                                  hop_length=feat_config.hop_length,
                                                  n_mels=feat_config.mel.n_mels,
                                                  fmax=feat_config.mel.fmax)
    elif feat_id == 'mel':
        features = librosa.feature.melspectrogram(y=y, sr=feat_config.sample_rate,
                                                  hop_length=feat_config.hop_length,
                                                  n_mels=feat_config.mel.n_mels,
                                                  fmax=feat_config.mel.fmax)
        features = np.log1p(np.abs(features)**2)                                                
    elif feat_id == 'chroma':
        features = librosa.feature.chroma_cqt(y=y, sr=feat_config.sample_rate, fmin=27.5, n_octaves=8, hop_length=feat_config.hop_length)

    beat_frames = []
    return features, beat_frames


def madmom_beats(audiofile, audio_duration):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audiofile)
    beat_times = np.asarray(proc(act))
    if beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0)
    if beat_times[-1] < audio_duration:
        beat_times = np.append(beat_times, audio_duration)
    beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beats


def get_features(audio_file, feat_id, feat_config, feat_type, y=None,
                 save=True):
    file_struct = FileStruct(audio_file)
    if valid_feat_files(file_struct, feat_id, feat_config):
        features = np.load(file_struct.get_feat_filename(feat_id), mmap_mode='r')
    else:
        if y is None:
            y, _ = librosa.load(audio_file, sr=feat_config.sample_rate)
        duration = librosa.get_duration(y=y, sr=feat_config.sample_rate)
        features, beat_frames = compute_features(y, feat_id, feat_config)
        if not os.path.isfile(FileStruct(audio_file).beat_file):
            compute_beats(audio_file, y=y, feat_config=feat_config)
        if save:
            duration = librosa.get_duration(y, sr=feat_config.sample_rate)
            write_features(features, file_struct, feat_id, feat_config, beat_frames, duration)
    return features


def compute_beats(audio_file, y, feat_config):
    file_struct = FileStruct(audio_file)
    duration = librosa.get_duration(y, sr=feat_config.sample_rate)
    beat_frames = madmom_beats(audio_file, duration)
    update_beats(file_struct, feat_config, beat_frames, duration)
    return 0


def validate_all_files(file_struct, feat_ids, feat_config):
    """ The only purpose of this function is to avoid opening the audio
    file multiple times in a row.
    """
    for feat_id in feat_ids:
        valid = valid_feat_files(file_struct, feat_id, feat_config)
        if valid is False:
            return False
    return valid


def process_track(audio_file, feat_ids, feat_config):
    file_struct = FileStruct(audio_file)
    print('Processing file:', audio_file)
    if not validate_all_files(file_struct, feat_ids, feat_config):
        print('Calculating features')
        y, _ = librosa.load(audio_file, sr=feat_config.sample_rate)
        for feat_id in feat_ids:
            get_features(file_struct.audio_file, feat_id, feat_config, 'framesync', y=y, save=True)
    else:
        print('All feature files already found, skipping.')
    return 0


def main(args):
    tracks = librosa.util.find_files(args.ds_path.joinpath('audio'))
    pool = mp.Pool(mp.cpu_count())
    funclist = []
    for file in tqdm(tracks):
        f = pool.apply_async(process_track, [file, args.feat_ids, config])
        funclist.append(f)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", help="Path to the dataset", type=Path,
                        default=Path('.'))
    parser.add_argument("--feat_ids", nargs='+', type=list,
                        default=['mel'], help='ID of features to save.')
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                        help="Number of jobs. Defaults to every cpus.")
    args, _ = parser.parse_known_args()
    config.no_cuda = True
    main(args)
