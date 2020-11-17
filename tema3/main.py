import gzip
import pickle
import numpy as np
from neural_net import NeuralNet

with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

if __name__ == "__main__":
    nn = NeuralNet([784, 100, 10], [i for i in range(10)], optimize_weights_init=True)
    # nn.train(train_set, learning_rate=0.01, batch_size=100, iterations=1)
    print(nn.test_accuracy(valid_set))
