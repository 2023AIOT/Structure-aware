# Structure-Aware Curriculum Transfer Learning for Cross-Domain Steel Surface Defect Detection

## Overview

This repository provides the implementation and experimental resources for the paper:

**"Structure-Aware Curriculum Transfer Learning for Cross-Domain Steel Surface Defect Detection"**

The proposed framework aims to improve cross-domain steel surface defect detection by combining:

- Structure-aware parameter transfer;
- Similarity-based source-domain sample selection;
- High-to-low curriculum transfer;
- Cross-domain generalization analysis on YOLOv5s and YOLOv8s.

The repository contains the training code, curriculum-learning configuration, similarity-ranking results, source-domain pretrained models, visualization scripts, and auxiliary experimental resources used in our study.

---

## Repository Structure

```text
Structure-aware/
│
├── README.md
│
├── curriculum.yaml
│
├── train.py
│
├── train_v8.py
│
├── build_similarity_subset_yolov5.py
│
├── build_similarity_subset_yolov8.py
│
├── build_curriculum_subsets.py
│
├── build_curriculum_subsets_yolov8.py
│
├── visualize_target_tsne.py
│
├── select_failure_cases.py
│
├── GC10_pretrained.pt
│
├── NEU_pretrained.pt
│
├── GC10_to_NEU_similarity.csv
│
├── NEU_to_GC10_similarity.csv
│
├── GC10_to_NEU_sim_top75.txt
└── NEU_to_GC10_sim_top75.txt
