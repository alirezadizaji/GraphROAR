# GraphROAR
At the moment, several explainer methods have been developed to perceive the functionality of graph neural networks. In this repo, we propose a new retraining approach to measure and compare these explainer methods by either keeping (GraphKAR) or removing (GraphROAR) most informative edges and thereafter checking their impact using retraining steps.
### Contents:
  - [Overview](#overview)
    - [Baseline](#baseline)
    - [Explanation](#explanation)
    - [Retraining](#retraining)
  - [Data](#data)
  - [Configuration](#configuration)
  - [Run](#Run)

## Overview
All running entry points are located at `src/GNN_Explainability/entrypoints/` and In general, there are three groups of running:
### Baseline
This includes training graph neural networks, basically graph convolution and graph inception networks, to be later perceived by explainers. Their relative addresses follow this pattern `train_gcn.[dataset].base.[gcn3l or gin3l]`.
### Explanation
The explainers will provide edge weightings per instance for retraining experiments. Despite others, SubgraphX does not provide probability edge weightings. Instead, we introduced a trick in which for each of five percentages proposed in retraining stage (10, 30, 50, 70, and 90), a separate set of binary edge masks is stored. their relative addresses follow this pattern: `explain_gcn.[dataset].[gcn3l or gin3l].[explainer]`.
### Retraining
Now to measure and compare explainer methods, probability edge weightings provided by these explainers are used to either keep (GraphKAR) or remove (GraphROAR) most informative edges and thereafter check their impact using the retraining step. Their relative addresses follow this pattern `train_gcn.[dataset].[kar or roar].[gcn3l or gin3l].[explainer]`.

## Data
At first, within the root directory of this repo, create a `data` folder. then download BA-2Motifs from [here](https://drive.google.com/file/d/134We2cb2PjoY1b6-k8KLmfviM0M4CEkT/view?usp=sharing), BA-3Motifs from [here](https://drive.google.com/drive/folders/1ZGrosPKm85phN54tSGl7-lmQFx-w4NTd?usp=sharing), and the rest from [TUDataset](https://drive.google.com/file/d/134We2cb2PjoY1b6-k8KLmfviM0M4CEkT/view?usp=sharing). Except for BA-2Motifs, the folder hierarchy of the rest are identical; which should be the same with the following:
```
data/
    ba_2motifs/
        processed/
           data.pt
    BA3Motifs/
        raw/ 
            BA3Motifs_A.txt
            BA3Motifs_graph_indicator.txt
            BA3motifs_graph_labels.txt
    ENZYMES/
        raw/
            ...
    IMDB-BINARY/
        raw/
            ... 
    MUTAG/
        raw/
            ...
```
For each dataset, the probability edge weights provided by each explainer during the second stage will be saved in the `data/[dataset]/explanation` folder.

## Configuration
Running configurations are located in `src/GNN_Explainability/config/`. The general configuration, including [baseline](#baseline) stage, is located at `base_config.py`, [explanation](#explanation) at `explanation/` and [retraining](#retraining) at `roar_config.py`; To know in detail about the attributes within each one, check out the documentation provided below of each attribute. 

## Run
In order to run each of the three groups mentioned in [overview](#overview), first change your directory to `src/`, then enter this command: `python main.py [seed number] [pattern of one of these three groups]`. For instance, to run GraphKAR on GradCAM explanation using GCN3l network and BA-2Motifs dataset, this command is required: `python main.py 12345 train_gcn.ba2motifs.kar.gcn3l.gradcam`. All seed numbers we used are listed in `src/main.py`.  
