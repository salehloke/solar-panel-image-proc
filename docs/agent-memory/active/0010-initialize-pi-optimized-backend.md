---
type: task
title: Initialize Pi-Optimized ML Backend
date: 2025-12-30
status: In Progress
tags:
  - RaspberryPi
  - FastAPI
  - Scikit-Learn
  - ComputerVision
  - HOG
  - GLCM
project: solar-panel
---

# Task: Initialize Pi-Optimized ML Backend

## Objective
Generate the FastAPI structure and the Python script for GLCM and HOG feature extraction, optimized for deployment on a Raspberry Pi.

## Context
- **Hardware**: Raspberry Pi (Resource constrained).
- **Goal**: Classify Dust, Bird Droppings, and Moss.
- **Methods**: SVM/Random Forest with manual feature extraction (HOG/GLCM).

## Visual Flow (Mermaid)
```mermaid
graph TD
    Image[Pi Camera Image] --> Resize[Resize to 128x128]
    Resize --> Gray[Grayscale Conversion]
    Gray --> HOG[HOG Extraction: Edge/Shape]
    Gray --> GLCM[GLCM Extraction: Texture]
    HOG --> Combined[Feature Vector Concatenation]
    GLCM --> Combined
    Combined --> SVM[SVM Inference]
    SVM --> API[FastAPI JSON Response]
```

## Plan
- [x] Create directory structure for the new edge backend.
- [ ] Implement `features.py` using `skimage` for optimized HOG/GLCM extraction.
- [ ] Implement `main.py` with FastAPI `/detect` endpoint.
- [ ] Create `requirements-pi.txt` excluding heavy dependencies like PyTorch for the edge node.

## Execution Log
- [x] Task created.
- [ ] Implementing feature extraction logic.
- [ ] Implementing FastAPI boilerplate.
