import matplotlib.pyplot as plt
import numpy as np
import torch


def visualization(**images):
    """ Visualizes the images
    Args:
        images (named arg - img sequence): - named argument and image sqeuence to visualize
    Example:
        img_visualization(img1 = img1, img2 = img2 ...)
    """

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
