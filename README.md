# Performance Evaluations of Computer Networks with Graph Neural Networks

This repository contains the slides and code of the _"Performance Evaluations of Computer Networks with Graph Neural Networks"_ presentation given at the [KuVS - AI in Networking Summer School 2022](https://arizk.github.io/AINet22/2022/index.html). See the `slides.pdf` file for the slides presented during the summer school.

The main goal is to train a Graph Neural Network (GNN) for predicting latency bounds in networks.
From a ML perspective, this is a regression task.
The steps below explain how to setup the running environment, get and prepare the dataset, and finally train a GNN for predicting latencies in networks.

**Warning:** The main goal of this repository is to be a tutorial. This code only represents a basic running sketch on how to work with GNNs. For more concrete and relastic applications, some parts need to be adapted.


## Get the code

The first step is to get the code with git:
```
$ git clone https://github.com/fabgeyer/ainet-summer-school-2022.git
$ cd ainet-summer-school-2022
```


## Setup your Python environment

Install the required python packages using `pip`:
```
$ pip install numpy pbzlib networkx nni scikit-learn tqdm
```

Install pytorch and pytorch-geometric following their respective guides:
- [Installation instructions for pytorch](https://pytorch.org/get-started)
- [Installation instructions for pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

The following versions were used for the tutorial:
```
networkx==2.6.3
nni==2.6
numpy==1.22.2
pbzlib==0.20211124
scikit-learn==1.0.2
torch==1.10.2
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
tqdm==4.62.3
```

## Get and precompute the dataset

The dataset from the [RTAS 2021 paper](https://doi.org/10.1109/RTAS52030.2021.00021) will be used.
Full instructions about the dataset are provided [here](https://github.com/fabgeyer/dataset-rtas2021).

The following command lines need to be run to download the files (approx. 400M will be downloaded):
```
$ wget -r ftp://m1596901:m1596901@dataserv.ub.tum.de/
$ mv dataset.ub.tum.de dataset
```

Once the dataset is downloaded, graphs need to be precomputed and saved:
```
$ python main.py
Parse the dataset and precompute the graphs
Process dataset/dataset-train.pbz: 11748it [01:31, 127.82it/s]
Precomputed graphs are saved as: dataset-train.pt
```


## Run the training

Once the dataset has been downloaded and prepared, the Graph Neural Network (GNN) can be trained.

### Single training run

To perform a single training run:
```
$ python main.py
```


### Hyper-parameter optimization using NNI

To perform hyper-parameter optimization, [NNI](https://nni.readthedocs.io/) is used.
Adapt the `nni-config.yml` file according to your machine and number of GPUs.
Then start `nni` as follows:
```
$ nnictl create --config nni-config.yml
```

Please consult the [NNI](https://nni.readthedocs.io/) documentation for further information on how to use NNI.
