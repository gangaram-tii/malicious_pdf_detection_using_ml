# Malicious PDF detection using ML

This repository contains a Python-based project for detecting malicious PDF files. It leverages machine learning techniques to identify potential threats embedded within PDF documents.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Future Work](#future_work)

## Introduction

PDF documents are widely used in both personal and professional communication due to it’s portability and readability. PDF files have several features that attackers can exploit to deliver malware or can carry out phishing attacks. The `Malicious PDF detection` project aims to enhance security by identifying potential threats within PDF files. PDF documents can be exploited by malicious actors to deliver malware, and this tool helps in detecting such threats before they can cause harm.

## Features

- **Machine Learning Models**: Utilizes various machine learning models to classify PDFs as malicious or benign.Installation

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/gangaram-tii/malicious_pdf_detection_using_ml.git
   cd malicious_pdf_detection_using_ml
   ```
2. **Create a Virtual Environment**  (Non NixOS):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. Setup development shell (For NixOS):

   ```bash
   nix-shell
   ```

## Usage

### Training:

1. **Extract features:**

   ```bash
   cd dataset
   python extract-features.py
   ```
2. Train Models:

   ```bash
   cd ../
   python knn_classifier.py
   python simple_nn_classifier.py
   python simple_nn_classifier.py
   ```

### Results:

#### **KNN Classifier:**

    Accuracy: 0.9903

    Confusion Matrix:
    [[2256   12]
    [  56 4676]]

    Key Metrics:
    [Precision, Recall, Specificity]
    [0.9974,    0.9882, 0.9947]



#### **Simple Feed forward NN:**

    Accuracy: 0.9913

    Confusion Matrix:
    [[2284   25]
    [  36 4655]]

    Key Metrics:
    [Precision, Recall,  Specificity]
    [0.9947,    0.9923, 0.9892]


#### XGBoost

    Accuracy:** 0.9947

    Confusion Matrix:
    [[2252   16]
    [  21 4711]]

    Key Metrics:
    [Precision, Recall,   Specificity]
    [0.9966,     0.9956,  0.9929]


## Future work

1. Train on more dataset
2. Classify a pdf based on result from all three models (final prediction based on 2 out of 3 result)
3. Provide unified interface to classify a single file to ease integration.
