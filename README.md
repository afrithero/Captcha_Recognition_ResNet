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

## Coming Soon
- Training script and dataset preparation
- Evaluation metrics
- Model optimization and deployment guide

Stay tuned for updates!