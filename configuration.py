# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np


class SubConfig():
    def __init__(self, parameters):
        for parameter_key in parameters:
            setattr(self, parameter_key, parameters[parameter_key])

    def __repr__(self):
        variables = vars(self)
        text = ''
        for var in variables:
            text += '    {:20}{}  \n'.format(var, variables[var])
        return text


class Config():
    def __init__(self, parameters):
        for parameter_key in parameters:
            setattr(self, parameter_key, parameters[parameter_key])

    def add_subconfig(self, subconfig_name, subconfig):
        setattr(self, subconfig_name, subconfig)

    def __repr__(self):
        variables = vars(self)
        text = ''
        for var in variables:
            if type(variables[var]) is SubConfig:
                text += '{}  \n'.format(var)
                text += str(variables[var])
            else:
                text += '{:24}{}  \n'.format(var, variables[var])
        return text

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
default_config_dict = {
    'default_bound_id': 'sf',  # Default boundary detection algorithm ("sf", "cnmf", "foote", "olda", "scluster", "gt")
    'default_label_id': None,  # Default label detection algorithm (None, "cnmf", "fmc2d", "scluster")
    'sample_rate': 22050,  # Default Sample Rate to be used.
    'n_fft': 2048,  # FFT size
    'hop_length': 256,  # Hop length in samples
    }

# Files and dirs
default_config_dict['results_dir'] = "results"

config = Config(default_config_dict)

# Dataset files and dirs
dataset_subconfig_dict = {
    'audio_dir': "audio",
    'estimations_dir': "estimations",
    'features_dir': "features",
    'references_dir': "references",
    'audio_exts': ['wav', 'mp3', 'aiff'],
    'estimations_ext': ".jams",
    'features_ext': ".json",
    'references_ext': ".jams"
}
dataset_subconfig = SubConfig(dataset_subconfig_dict)
config.add_subconfig('dataset', dataset_subconfig)

# Spectrogram
spectrogram_config_dict = {
    'ref_power': 'max'}
spectrogram_config = SubConfig(spectrogram_config_dict)
config.add_subconfig('spectrogram', spectrogram_config)

# Constant-Q transform
cqt_config_dict = {
    'bins': 120,
    'norm': np.inf,
    'filter_scale': 1.0,
    'ref_power': 'max',
    }
cqt_config = SubConfig(cqt_config_dict)
config.add_subconfig('cqt', cqt_config)

# Melspectrogram
mel_config_dict = {
    'n_mels': 60,  # Number of mel filters
    'ref_power': 'max',  # Reference function to use for the logarithm power.
    'fmax': 8000
    }


mel_config = SubConfig(mel_config_dict)
config.add_subconfig('mel', mel_config)


mfcc_config_dict = {
    'n_mels': 60,  # Number of mel filters
    'ref_power': 'max',  # Reference function to use for the logarithm power.
    'fmax': 8000
    }


mfcc_config = SubConfig(mfcc_config_dict)
config.add_subconfig('mfcc', mfcc_config)

# Chomagram
chromagram_config_dict = {
    'norm': np.inf,
    'ref_power' : 'max'
    }

harmonic_config_dict = {}
harmonic_config = SubConfig(harmonic_config_dict)
config.add_subconfig('harmonic', harmonic_config)

chromagram_config = SubConfig(chromagram_config_dict)
config.add_subconfig('chroma', chromagram_config)

# Embed features
embedding_feat_config_dict = {'base_feat_id': 'cqt',
                              'n_embedding': 512,  # Number of feature frames in the learned embedding,
                              'embed_hop_length': 86  # Embedding hop length in feature frames
                              }
embedding_config = SubConfig(embedding_feat_config_dict)
config.add_subconfig('embedding', embedding_config)

# experiment_config = config

# Dataset parameters
config.model_name = None
config.ds_path = None
config.listb = None
# config.exclude_silence = True
config.training_strategy = 'supervised' 

# Files parameters
config.checkpoints_dir = Path('results').joinpath('checkpoints')
config.models_dir = Path('results').joinpath('models')
config.tensorboard_logs_dir = Path('results').joinpath('runs')

# Model parameters
config.conv_kernel_size = (3, 3)

# Default feature parameters
config.feat_id = 'mel'
config.feat_type = 'beat_sync'


# Training parameters
config.epochs = 250
config.batch_size = 512
config.learning_rate = 1e-4
config.resume = False
config.test = False
config.architecture = 'EmbedNet'
config.margin = 0.1
config.no_cuda = False
config.quiet = False
config.nb_workers = 8
config.seed = 42
config.seg_algo = 'scluster'