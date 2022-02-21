# ABAW3 CHALLENGE

## Table of Contents
+ [Overview](#Overview)
+ [Team information](#team-information)

## Overview
+ **Subject**: CVPR 2022: 3rd Workshop and Competition on Affective Behavior Analysis in-the-wild (ABAW)
+ **Home page**: https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/
+ **Description**: The ABAW Workshop and Competition has a unique aspect of fostering cross-pollination of different disciplines, bringing together experts (from academia, industry, and government) and researchers of mobile and ubiquitous computing, computer vision and pattern recognition, artificial intelligence and machine learning, multimedia, robotics, HCI, ambient intelligence and psychology. The diversity of human behavior, the richness of multi-modal data that arises from its analysis, and the multitude of applications that demand rapid progress in this area ensure that our events provide a timely and relevant discussion and dissemination platform.

+ **Tasks**:
  1) Valence-Arousal (VA) Estimation Challenge
      - Concordance Correlation Coefficient (CCC) of valence and arousal: P = 0.5 * (CCC_arousal + CCC_valence)
  2) Expression (Expr) Classification Challenge
     - Average F1 Score across all 8 categories: P = ∑(F1)/8
  3) Action Unit (AU) Detection Challenge
     - Average F1 Score across all 12 categories: P = ∑ (F1) / 12
  4) Multi-Task-Learning (MTL) Challenge
     - P = 0.5 * (CCC_arousal + CCC_valence) + 0.125 * ∑ (F1_expr) + ∑ (F1_au) / 12

## Team information: 
+ Team Name: **BEAST**
+ Team Members:
  + **Vo Minh Cong** congvm.it@gmail.com
+ Affiliation: Chonnam National University, South Korea


## Coding

+ Dataset: /mnt/DATA2/congvm/Affwild2 
+ Annotations: /mnt/DATA2/congvm/Affwild2/Annotations

```
python 0.preprocess-csv.py
python 1.train.py
```

## Current Results

| Backbone |  Models  | $CCC_{v}@CCC_{a}@P_{VA}$ | $P_{exp}$ | $P_{au}$ | $P_{mtl}$ |
| :------: | :------: | :----------------------: | :-------: | :------: | :-------: |
| VGGFace  | Baseline |      0.31@0.17@0.24      |   0.23    |   0.39   |   0.86    |
|          |          |                          |           |          |           |
|          |          |                          |           |          |           |