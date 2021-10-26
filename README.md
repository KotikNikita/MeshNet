

## MeshNet: Mesh Neural Network for 3D Shape Representation
This repository is not an original official implementation of the work, but a refactored codebase of https://github.com/iMoonLab/MeshNet. Performed within the FSE coursework at Skoltech.



### Installation

First you have to clone repository:
```
git clone https://github.com/KotikNikita/MeshNet
```
Run docker, you have 2 options:
1) From DockerHub:
```
docker pull poliik/meshnet_docker:latest
docker run -it poliik/meshnet_docker:latest
```
2) Build image locally:
```
docker build -t meshnet_docker -f Dockerfile .
docker run -it meshnet_docker
```

### Usage

##### Data Preparation

Firstly, you should download the [reorganized ModelNet40 dataset](https://drive.google.com/open?id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz). 
To download train dataset, run following commnd
```
bash download.sh
```

##### To train model, run:
```
bash train.sh
```


##### To test model, run:
```
bash test.sh
```

### Modifications
For each data file `XXX.off` in ModelNet, we reorganize it to the format required by MeshNet and store it into `XXX.npz`. The reorganized file includes two parts of data:

* The "face" part contains the center position, vertices' positions and normal vector of each face.
* The "neighbor_index" part contains the indices of neighbors of each face.

If you wish to create and use your own dataset, simplify your models and organize the `.off` files similar to the ModelNet dataset. 
Then use the code in `data/preprocess.py` to transform them into the required `.npz` format. 
Notice that the parameter `max_faces` in config files should be maximum number of faces among all of your simplified mesh models. 


You can modify the configuration in the `config/train_config.yaml` for your own training, including the CUDA devices to use, the flag of data augmentation and the hyper-parameters of MeshNet.

The pretrained MeshNet model weights are stored in [pretrained model](https://drive.google.com/open?id=1l8Ij9BODxcD1goePBskPkBcgKW76Ewcs). You can download it and configure the "load_model" in `config/test_config.yaml` with your path to the weight file.
