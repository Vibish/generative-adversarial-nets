# Imports
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_mldata
from ganetwork import CGAN

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home='.')
X, y = mnist.data, mnist.target

# CGAN
cgan = CGAN(
    n_Z_features=128, 
    discriminator_hidden_layers=[(100, tf.nn.relu)], 
    generator_hidden_layers=[(100, tf.nn.relu)]
    )

# Train CGAN and plot/save 10 generated sample images per epoch for class_label=4
if __name__ == '__main__':
    cgan.train(
        X,
        y, 
        nb_epoch=10000, 
        batch_size=128, 
        logging_options=['plot_images', 'save_images'], 
        logging_steps=1, 
        n_samples=10,
        class_label=4
        )