## [Scene Graphs with Permutation-Invariant Structured Prediction](https://arxiv.org/abs/1802.05451)

## About this repository
This repository contains an implementation of our best variant (Linguistic) of the Scene Graph Prediction (SGP) model introduced in the paper [Scene Graphs with Permutation-Invariant Structured Prediction](https://arxiv.org/abs/1802.05451).
Specifically, the repository allow to run scene-graph classification (recall@100) evaluation script on our pre-trained model or alternatively (1) train an SGP model (2) evaluate the trained model using scene-graph classification (recall@100) evaluation script.

## Introduction
In Scene-Graph classification task, the input is an image annotated with a set of rectangles that bound entities in the image, known as bounding boxes.
The goal is to label each bounding box with the correct entity category, and every pair of entities with their relation,
such that they form a coherent graph, known as a scene graph. In a scene graph, nodes correspond to bounding-boxes labeled with the entity category and edges correspond to relations among entities.

## About the implemented model
Our model has two components: A Label Predictor (LP) that takes as input an image with bounding boxes and outputs a distribution over labels for each entity and relation.
Then, a Scene Graph Predictor (SGP) that takes all label distributions and predicts more consistent label distributions jointly for all entities and relations. SGP satisfies the graph permutation invariance property intoduced in the paper.
The repository includes just the SGP model which gets as input a label distribution that created a head by our LP over [VisualGenome dataset](https://visualgenome.org).
The model implemented using [TensorFlow](https://www.tensorflow.org/)

## Dependencies
To get started with the framework, install the following dependencies:
- [Python 2.7](https://www.python.org/)
- [tensorflow-gpu 1.0.1](https://www.tensorflow.org/)
- [matplotlib 2.0.2](http://matplotlib.org/)
- [h5py 2.7.0](http://www.h5py.org/)
- [numpy 1.12.1](http://www.numpy.org/)
- [pyyaml 3.12](https://pypi.python.org/pypi/PyYAML)

Run `"pip install -r requirements.txt"`  - to install all the requirements.

## Instructions
1. Run `"python Run.py download"` to download and extract train, validation and test data (the data includes the initial label distribution that created a head by our LP)
2. Run `"python Run.py eval gpi_ling_orig_best <gpu-number>"` to evaluate the pre-trained model (recall@100 SG Classification).
3. Run `"python Run.py train gpi_ling_new <gpu-number>"` to train a new model.
4. Run `"python Run.py eval gpi_ling_new_best <gpu-number>"` to evaluate the new model. (recall@100 SG Classification).


### If you find this project helpful in your work, please cite  [this paper](https://arxiv.org/abs/1802.05451).

