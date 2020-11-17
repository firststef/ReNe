import gzip
import pickle
import unittest
import numpy as np
from neural_net import NeuralNet

with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

if __name__ == "__main__":
    nn = NeuralNet([2, 2, 1], [0], dont_update_weights=True)
    nn.layers_w = [
        np.array([[-3, 6], [1, -2]], dtype=float),
        np.array([[8], [4]], dtype=float),
        np.array([], dtype=float)
    ]
    nn.train([[[2, 6]], [1]], learning_rate=0.5, batch_size=1, iterations=1, save=False)
    print(nn.layers_w)
