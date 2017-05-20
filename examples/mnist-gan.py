# Imports
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_mldata
from ganetwork import GAN

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home='.')
X, y = mnist.data, label_binarize(mnist.target, classes=np.unique(mnist.target))

# GAN
gan = GAN(
    n_Z_features=128, 
    discriminator_hidden_layers=[(100, tf.nn.relu)], 
    generator_hidden_layers=[(100, tf.nn.relu)]
    )

# Train GAN and plot/save 10 generated sample images per epoch
if __name__ == '__main__':
    gan.train(
        X, 
        nb_epoch=10000, 
        batch_size=128, 
        logging_options=['plot_images', 'save_images'], 
        logging_steps=1, 
        n_samples=10
        )