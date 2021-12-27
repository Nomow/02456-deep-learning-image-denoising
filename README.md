# Autoencoders for image quality improvement
> A project for DTU 02456 Deep Learing course on the topic of autoencoder neural network architecture used in image denoising tasks.

## About
The goal of the project was to explore the available architectures that could be used for removing noise in images. The initial choice was the autoencoder architecture, as it is a widely used in these types of problems. Further explorations showed that the pre-trained architectures of U-Net and U-Net++ can be easily adapted to outperform typical or modified autoencoder architectures.

First tests were performed using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset with artificially added noise, but the primary goal was to attempt removing noise from dermatiological images. Dermatological images dataset was acquired by cooperation with a company [Omhu](https://omhu.com/) and due to confidentiality, most examples of the usage of this dataset were removed

Project implemented primarily in [Pytorch](https://pytorch.org/), artificial noise was generated using [Albumentations](https://albumentations.ai/) library.

## Getting started

To evaluate the results of the project, a notebook **Results.ipynb** was prepared. There, one can find the architectures implemented in the project, a link to a google drive with the trained model weights and  a presentation of achieved results. It is also possible to rerun sectons of the code related to CIFAR-10 dataset. The dermatological dataset sections contain only a presentation of the models and training loss function plots, no dataset usage examples due to confidentiality. 

