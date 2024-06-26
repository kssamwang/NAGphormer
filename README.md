# NAGphormer
This is the code for our ICLR 2023 paper 
**NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs**.

![NAGphormer](./NAGphormer.jpg)

## Requirements
Python == 3.8

Pytorch == 1.11

dgl == 0.9

CUDA == 11.3

```sh
conda create -n gf1 python=3.8 -y
conda activate gf1
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 2014  pip install packaging==20.0 pyparsing==2.3.1 python-dateutil==2.7
pip install matplotlib==3.7.3
wget https://conda.anaconda.org/dglteam/linux-64/dgl-cuda11.3-0.9.0-py38_0.tar.bz2
conda install dgl-cuda11.3-0.9.0-py38_0.tar.bz2
rm -f dgl-cuda11.3-0.9.0-py38_0.tar.bz2
```

## Usage

You can run each command in "commands.txt".

You could change the hyper-parameters of NAGphormer if necessary.

Due to the space limitation, we only provide several small datasets in the "dataset" folder.

```sh
unzip dataset/dataset.zip -d .
```

For small-scale datasets, you can download them from https://docs.dgl.ai/tutorials/blitz/index.html.

For large-scale datasets, you can download them from https://github.com/wzfhaha/GRAND-plus.


## Cite
If you find this code useful, please consider citing the original work by authors:
```
@inproceedings{chennagphormer,
  title={NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs},
  author={Chen, Jinsong and Gao, Kaiyuan and Li, Gaichao and He, Kun},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2023}
}
```
