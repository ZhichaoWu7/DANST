# DANST: a Deep Domain Adversarial Neural Network based approach for Cell-type Deconvolution in Spatial Transcriptomics



## Overview
Spatial transcriptomics is an emerging technology that can analyze gene expression profiles of tissues while preserving spatial location information. However, current spatial transcriptomics technologies are difficult to achieve single-cell resolution. In order to restore the relative proportion of each cell type in the sample from mixed gene expression data, this paper proposes a general framework based on deep domain adversarial neural network (DANST). We use single-cell RNA sequencing data with clear cell type labels but no spatial location information to construct pseudo spatial transcriptome data, calculates pseudo coordinates through the feature similarity of pseudo data and real data. And then, we optimize features with the assistance of variational auto-encoder, and introduces domain adversarial architecture to connect the feature distributions of real data and pseudo data, transferring cell type labels from pseudo data to real data. Experiments on multiple datasets show that DANST can have a better deconvolution effect on spatial transcriptome data from different species and sources, verifying that this method has broad application prospects in the fields of tumor microenvironment interpretation and clinical diagnosis and analysis.

## Requirements
You'll need to install the following packages in order to run the codes.
* python>=3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy>=1.22.3
* pandas>=1.5
* scanpy==1.11.0
* leiden==0.10.2
* igraph==0.11.8
* scipy==1.5.4
* scikit-learn==0.24.2
* matplotlib==3.4.2

## Getting started
See Main.py and Tutorial for DANST.ipynb

## Software dependencies
scanpy

pytorch

pyG