import gzip
import pickle
import numpy as np
from neural_net import NeuralNet
from neural_net_activations import softmax

with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

if __name__ == "__main__":
    nn = NeuralNet([784, 100, 10], [i for i in range(10)], last_activation=softmax, optimize_weights_init=True)
    nn.train_optimized(train_set, learning_rate=0.1, batch_size=10, iterations=10, test_data=valid_set, save=True)
    print(nn.test_accuracy(valid_set))
