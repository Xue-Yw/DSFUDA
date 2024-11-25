# DSFUDA

This repository contains the implementation of our method based on the Detectron2 framework.


## 1. Environment Setup
Ensure the following dependencies:
- PyTorch 1.10.0
- Python 3.8 (Ubuntu 20.04)
- CUDA 11.3

Install PyTorch and related libraries

`pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0`

Install Detectron2 framework

`python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html`

Install additional dependencies

`pip install -r requirements.txt`

## 2. Dataset Preparation
Download and prepare datasets:
- CDTBS Dataset: https://github.com/CVIU-CSU/ComparisonDetector

Place in datasets/VOC2012cdtbs4class

- SIPaKMeD Dataset: https://www.cs.uoi.gr/~marina/sipakmed.html

Place in datasets/VOC2012

## 3. Train the Models
Train the Source Domain Model

`nohup python tools/train_net.py --num-gpus 1 --config-file configs/train/source.yaml >nohup_out/source.out 2>&1 &`

Train the Target Domain Model using the source model

`nohup python tools/train_net.py --num-gpus 1 --model-dir source_model/source/model_final.pth --config-file configs/train/DSFUDA.yaml >nohup_out/DSFUDA.out 2>&1 &`
