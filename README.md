# Neuro-Face-Attention.

PyTorch implementation of Neuro-Face-Attention by [Arun Das](https://github.com/arundasan91), [Mehrad Jaloli](https://github.com/mehradjaloli), and [Paul Rad](https://github.com/paulNrad).

We tackle the problem of learning low-level facial attributes to encode muscle movements as a dense vector for stuttering studies.

## Setup
The code is developed and tested on `Python3.6`. Required packages are listed in `requirements.txt` file.

## Dataset
Raw videos maybe passed as a parameter to the dataset preprocessing scripts. There are two scripts provided:

1. To find the target sequence in the video, and save the results in a `JSON` file.
2. To segment the input video according to the generated `JSON` file for all subjects.

Dataset is required to be in the following folder structure:
```
Dataset Root
    Subject 1
        Study 1
        Study 2
        ...
        Study n
    Subject 2
        Study 1
        Study 2
        ...
        Study n
    ...
    Subject n
```

#### Usage
Finding Target Sequence:
```bash
python3 find_cue_frames.py --source /home/user/path/to/dataset_root/
```

Labelling the videos:
```bash
python3 label_videos.py --source /home/user/path/to/dataset_root/
```

## Dependencies
- [OpenCV 3.4.2 from source](https://github.com/arundasan91/Server-and-Cloud-Essentials/blob/master/install_opencv.sh), including GStreamer and FFMPEG
- Matplotlib==3.0.3
- Numpy==1.17.1
