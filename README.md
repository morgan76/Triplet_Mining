# A Repetition-based Triplet Mining Approach for Music Segmentation
This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [A Repetition-based Triplet Mining Approach for Music Segmentation](https://hal.science/hal-04202766/) 
presented at ISMIR 2023.

The overall format based on the
[MSAF](https://ismir2015.ismir.net/LBD/LBD30.pdf) package. 

## Table of Contents
0. [Usage](#usage)
0. [Requirements](#requirements)
0. [Citing](#citing)
0. [Contact](#contact)

## Usage
The network can be trained with:

```
python trainer.py --feat_id {feature type} --ds_path {path to the dataset}
```

The dataset format should follow:
```
dataset/
├── audio                   # audio files (.mp3, .wav, .aiff)
├── features                # feature files (.npy)
└── references              # references files (.jams)
```

To segment tracks and save deep embeddings:
```
python segment.py --ds_path {path to the dataset} --model_name {trained model name} --bounds {return boundaries and segment labels}
```

## Requirements
```
conda env create -f environment.yml
```

## Citing
```
@inproceedings{buisson2023repetition,
  title={A Repetition-based Triplet Mining Approach for Music Segmentation},
  author={Buisson, Morgan and Mcfee, Brian and Essid, Slim and Crayencour, Helene-Camille},
  booktitle={International Society for Music Information Retrieval (ISMIR)},
  year={2023}
}
```

## Contact
morgan.buisson@telecom-paris.fr# Triplet_Mining
