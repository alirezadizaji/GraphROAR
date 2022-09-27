# GraphROAR
At the moment, several explainer methods have been developed to perceive the functionality of graph neural networks. In this repo, we propose a retraining approach to measure and compare these explainer methods by either keeping (GraphKAR) or removing (GraphROAR) most informative edges and thereafter check their impact using retraining steps.
### Contents:
  - [Overview](#overview)
  - [Data](#data)
  - [Run](#Run)

## Overview
All running entrypoints are located at `src/GNN_Explainability/entrypoints/` and In general, there are three groups of running:
- Baseline: This includes training graph neural networks, basically graph convolution and graph inception networks to be perceived by explainers. Their relative addresses follow this pattern `train_gcn.[dataset].base.[gcn3l or gin3l]`.
- Explanation: The explainers will provide edge weightings per instance for retraining experiments. Despite others, SubgraphX does not provide probability edge weightings. Instead, we introduced a trick in which for each of five percentages proposed in retraining stage (10, 30, 50, 70, and 90), a separate set of binary edge masks is stored. their relative addresses follow this pattern: `explain_gcn.[dataset].[gcn3l or gin3l].[explainer]`.
- GraphROAR and GraphKAR: Now to measure and compare explainer methods, probability edge weightings provided by explainers are used to either keep (kar) or remove (roar) most informative edges and thereafter check their impact using retraining step. Their relative addresses follow this pattern `train_gcn.[dataset].[kar or roar].[gcn3l or gin3l].[explainer]`.

## Data

## Run
In order to run each of these three groups, first change your directory to `src/`, then enter this command: `python main.py [seed number] [pattern of one of these three groups]`. For instance, in order to run GraphKAR on GradCAM explanation using GCN3l network and BA-2Motifs dataset, this command is required: `python main.py 12345 train_gcn.ba2motifs.kar.gcn3l.gradcam`. All seed numbers we used are listed in `src/main.py`.  