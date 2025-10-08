# Captcha Recognition with ResNet

This repository implements a deep learning-based Captcha recognition system using the ResNet architecture. The goal is to accurately predict the characters in Captcha images by leveraging the feature extraction power of convolutional neural networks.

The current implementation supports inference on Captcha images, and more details (including training pipeline, dataset information) will be added soon.
# Problem Description
The Captcha images contain a sequence of digits, and the model must learn not only to recognize individual digits but also to preserve their order, which is essential for correct interpretation.

We define three progressively challenging tasks:

- Task 1: Recognize a single digit in the image.

- Task 2: Recognize two digits in the image, where the order of digits matters.

- Task 3: Recognize four digits in the image, maintaining the correct sequential order.

An example is shown below, where the target output is '0751'. Each digit is highlighted and labeled to indicate its position in the sequence. The model must output the correct digit string by identifying the characters and their order from left to right.

![Logo](https://i.postimg.cc/fy9TB6cr/2025-07-30-3-56-10.png)

## Features
- ResNet-based architecture
- Supports multi-character Captcha prediction

---
## Environment Setup
```bash
pip install -r requirement.txt
```
## Data Preparation 
This project uses the **Captcha Hacker 2023 Spring** dataset from Kaggle.
### Step 1. 
- Go to your Kaggle account → Account → Create New API Token
- A file named kaggle.json will be downloaded
- Place it under ~/.kaggle/
### Step 2. 
```bash
kaggle competitions download -c captcha-hacker-2023-spring
unzip captcha-hacker-2023-spring.zip
```
The directory named dataset should be placed under the root level of this repository.

## Data Preprocessing

This project includes a reproducible preprocessing pipeline to (1) **augment** the training images and (2) **denoise** them for more robust training. The script is `src/data_preprocess.py`. It operates on three tasks stored under `dataset/train/{task1,task2,task3}`.
```bash
cd src
python data_preprocess.py
```

## Train the Model
```bash
cd src
python main.py --mode train
```

## Evaluate the Model
- Using a local checkpoint
```bash
cd src
python main.py --mode evaluate --checkpoint [your checkpoint path]
```
- Using a registered MLflow model
```bash
cd src
python main.py --mode evaluate --uri [your registered MLflow model uri]
```

## Predict
- Using a local checkpoint
```bash
cd src
python main.py --mode predict \
  --checkpoint [your checkpoint path] \
  --sample_submission [input csv path] \
  --submission [output csv path]

```

- Using a registered MLflow model
```bash
cd src
python main.py --mode predict \
  --uri [your registered MLflow model uri] \
  --sample_submission [input csv path] \
  --submission [output csv path]

```


