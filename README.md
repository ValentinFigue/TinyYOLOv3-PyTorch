# TinyYOLOv3 in PyTorch

This repositery is an Implementation of Tiny YOLO v3 in Pytorch which is lighted version of YoloV3, much faster and still accurate.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Structure](#structure)
- [Usage](#usage)
  * [Training](#training)
    + [Training on COCO dataset](#training-on-coco-dataset)
  * [Inference](#inference)
  * [Options](#options)
- [Note](#note)

## Introduction


You-Only-Look-Once (YOLO) and his lighted version TinyYOLOv3 was introduced by Joseph Redmon et al. a
Three versions were implemented in C, with the framework called [darknet](https://github.com/pjreddie/darknet) but no implementation of the lighted version has been made in Pytorch.

This repo implements the network model of TinyYOLOv3 in the PyTorch framework.
An implementation in Pytorch has already been made e.g. [westerndigitalcorporation/YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch).
This repositery developed the model from their work.

Both inference and training part are implemented but no original weights have been provided yet.

## Getting Started

Before cloning the repo to your local machine, make sure that `git-lfs` is installed. See details about `git-lfs`, see [this link](https://www.atlassian.com/git/tutorials/git-lfs#installing-git-lfs).

After `git-lfs` is installed. Run the following command to install the dependencies.

```
git lfs install
git clone https://github.com/ValentinFigue/TinyYOLOv3-PyTorch/
cd TinyYOLOv3-PyTorch
pip install -r requirements.txt

```

## Prerequisites

The repo is tested in `Python 3.7`. Additionally, the following packages are required:

```
numpy
torch>=1.0
torchvision
pillow
```



## Structure

The repo is structured as following as the original implementation of YoloV3 in Pytorch:
```
├── src
│   └── [source codes]
├── weights
│   ├── yolov3_original.pt
├── data
│   ├── coco.names
│   └── samples
├── fonts
│   └── Roboto-Regular.ttf
├── requirements.txt
└── README.md
```

`src` folder contains the source codes.
`weights` folder contains the original weight file trained by Joseph Redmon et al.
`data/coco.names` file lists the names of the categories defined in the COCO dataset.
`fonts` folder contains the font used by the Pillow module.


## Usage

### Training

No pre-trained weight have already been provided and so the model need to be trained from scratch.

#### Training on COCO dataset

To train on COCO dataset, first you have to download the dataset from [COCO dataset website](http://cocodataset.org/#home).
Both images and the annotations are needed.
Secondly, `pycocotools`, which serves as the Python API for COCO dataset needs to be installed.
Please follow the instructions on [their github repo](https://github.com/cocodataset/cocoapi) to install `pycocotools`.

After the COCO dataset is properly downloaded and the API setup, the training can be done by:

```
python3 main.py train --verbose --img-dir /path/to/COCO/image/folder --annot-path /path/to/COCO/annotation/file --reset-weights
```
You can see the network to converge within 1-2 epochs of training.

### Inference

To run inference on one image folder, run:

```
python3 main.py test --img-dir /path/to/image/folder --save-det --save-img
```

The `--save-det` option will save a `json` detection file to the output folder. The formate matches COCO detection format for easy benchmarking.
The `--save-img` option

### Options

`main.py` provides numerous options to tweak the functions. Run `python3 main.py --help` to check the provided options.
The help file is pasted here for your convenience. But it might not be up-to-date.

```
usage: main.py [-h] [--dataset DATASET_TYPE] [--img-dir IMG_DIR]
               [--batch-size BATCH_SIZE] [--n-cpu N_CPU] [--img-size IMG_SIZE]
               [--annot-path ANNOT_PATH] [--no-augment]
               [--weight-path WEIGHT_PATH] [--cpu-only] [--from-ckpt]
               [--reset-weights] [--last-n-layers N_LAST_LAYERS]
               [--log-dir LOG_DIR] [--verbose] [--debug] [--out-dir OUT_DIR]
               [--save-img] [--save-det] [--ckpt-dir CKPT_DIR]
               [--save-every-epoch SAVE_EVERY_EPOCH]
               [--save-every-batch SAVE_EVERY_BATCH] [--epochs N_EPOCH]
               [--learning-rate LEARNING_RATE] [--class-path CLASS_PATH]
               [--conf-thres CONF_THRES] [--nms-thres NMS_THRES]
               ACTION

positional arguments:
  ACTION                'train' or 'test' the detector.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_TYPE
                        The type of the dataset used. Currently support
                        'coco', 'caltech' and 'image_folder'
  --img-dir IMG_DIR     The path to the folder containing images to be
                        detected or trained.
  --batch-size BATCH_SIZE
                        The number of sample in one batch during training or
                        inference.
  --n-cpu N_CPU         The number of cpu thread to use during batch
                        generation.
  --img-size IMG_SIZE   The size of the image for training or inference.
  --annot-path ANNOT_PATH
                        TRAINING ONLY: The path to the file of the annotations
                        for training.
  --no-augment          TRAINING ONLY: use this option to turn off the data
                        augmentation of the dataset.Currently only COCO
                        dataset support data augmentation.
  --weight-path WEIGHT_PATH
                        The path to weights file for inference or finetune
                        training.
  --cpu-only            Use CPU only no matter whether GPU is available.
  --from-ckpt           Load weights from checkpoint file, where optimizer
                        state is included.
  --reset-weights       TRAINING ONLY: Reset the weights which are not fixed
                        during training.
  --last-n-layers N_LAST_LAYERS
                        TRAINING ONLY: Unfreeze the last n layers for
                        retraining.
  --log-dir LOG_DIR     The path to the directory of the log files.
  --verbose             Include INFO level log messages.
  --debug               Include DEBUG level log messages.
  --out-dir OUT_DIR     INFERENCE ONLY: The path to the directory of output
                        files.
  --save-img            INFERENCE ONLY: Save output images with detections to
                        output directory.
  --save-det            INFERENCE ONLY: Save detection results in json format
                        to output directory
  --ckpt-dir CKPT_DIR   TRAINING ONLY: directory where model checkpoints are
                        saved
  --save-every-epoch SAVE_EVERY_EPOCH
                        TRAINING ONLY: Save weights to checkpoint file every X
                        epochs.
  --save-every-batch SAVE_EVERY_BATCH
                        TRAINING ONLY: Save weights to checkpoint file every X
                        batches. If value is 0, batch checkpoint will turn
                        off.
  --epochs N_EPOCH      TRAINING ONLY: The number of training epochs.
  --learning-rate LEARNING_RATE
                        TRAINING ONLY: The training learning rate.
  --class-path CLASS_PATH
                        TINFERENCE ONLY: he path to the file storing class
                        label names.
  --conf-thres CONF_THRES
                        INFERENCE ONLY: object detection confidence threshold
                        during inference.
  --nms-thres NMS_THRES
                        INFERENCE ONLY: iou threshold for non-maximum
                        suppression during inference.
```


## Note

This work is not yet finished and may contain errors.