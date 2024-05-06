#!/bin/bash
conda create -n gf1 python=3.8
conda activate gf1
nvidia-smi
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 2014  pip install packaging==20.0 pyparsing==2.3.1 python-dateutil==2.7
pip install matplotlib==3.7.3
wget https://conda.anaconda.org/dglteam/linux-64/dgl-cuda11.3-0.9.0-py38_0.tar.bz2
conda install dgl-cuda11.3-0.9.0-py38_0.tar.bz2
rm -f dgl-cuda11.3-0.9.0-py38_0.tar.bz2

cd dataset
unzip dataset.zip -d .
rm dataset.zip
