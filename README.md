# Neuro-Face-Attention.

Tensorflow implementation of Neuro-Face-Attention by [Arun Das](https://github.com/arundasan91), [Henry Chacon](https://github.com/henrychacon), and [Paul Rad](https://github.com/paulNrad).

We tackle the problem of learning low-level facial attributes to encode muscle movements as a dense vector for stuttering studies.

## Setup
The code is developed and tested on `Python3.6`. Required packages are listed in `requirements.txt` file.

## Dataset
Raw videos maybe passed as a parameter to the dataset preprocessing scripts. However, the current data pipeline excepts face AU's as the input.

Dataset is required to be in the following folder structure for data processing scripts to work:
```
Dataset Root
    Subject 1
        Study 1
            Paradigm 1
            Paradigm 2
        Study 2
            Paradigm 1
            Paradigm 2
        ...
        Study n
            Paradigm 1
            Paradigm 2
    Subject 2
        Study 1
            Paradigm 1
            Paradigm 2
        Study 2
            Paradigm 1
            Paradigm 2
        ...
        Study n
            Paradigm 1
            Paradigm 2
    ...
    Subject n
```

#### Usage
There are several Jupyter Notebooks in the `src` folder. Please use the data pipeline file to pre-process your data. Model pipeline files have the deep learning architecture in them. Please use them to train the models. Each trained model will be saved in specified directories.

## Dependencies
- [OpenCV 3.4.2 from source](https://github.com/arundasan91/Server-and-Cloud-Essentials/blob/master/install_opencv.sh), including GStreamer and FFMPEG
- Matplotlib==3.0.3
- Numpy==1.17.1
