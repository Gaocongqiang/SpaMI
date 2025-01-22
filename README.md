# SpaMI：A graph neural network-based spatial multi-omics data integration model for deciphering spatial domains

## Overview
To integrate spatial transcriptome data more efficiently, we introduce the SpaMI method. It is an efficient and universally applicable deep learning method designed for the integrated representation of spatial multimodal data from the same tissue section. The method is able to take into account the heterogeneity of data from different histologies, preserve the biological significance of different histologies, and organically integrate the data so that the integrated data can reveal more comprehensive biological features, providing a comprehensive and complementary perspective for understanding cell expression patterns and spatial organization.

![](https://github.com/Gaocongqiang/SpaMI/blob/main/workflow.jpg)

## Dependencies
SpaMI has the following dependencies when running the codes.
* python==3.8
* h5py==3.10.0
* scikit-learn==1.3.2
* episcanpy==0.4.0
* scipy==1.10.1
* numpy==1.22.4
* pandas==2.0.3
* scanpy==1.9.8
* anndata==0.9.2
* torch==2.2.1
* torch_geometric==2.5.1
* matplotlib==3.7.5
* tqdm==4.66.2
* jupyter==1.0.0
* cudnn>=10.2

## Tutorial
We have curated tutorials to assist you in operating the SpaMI model. You can locate these tutorials in the SpaMI/Tutorial directory of the project. 

## Data
In this study, we applied SpaMI to multiple simulated and real spatial multi-omics data. All data are available from the following link. <https://figshare.com/articles/dataset/SpaMI_dataset/28059641>
