import gzip
import pickle
import unittest
import numpy as np
from neural_net import NeuralNet
from neural_net_activations import softmax

with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')


class TestTheSmallestNet(unittest.TestCase):
    def test_our_net(self):
        nn = NeuralNet([2, 2, 1], [0], last_activation=softmax, dont_update_weights=True)
        nn.layers_w = [
            np.array([[-3, 6], [1, -2]], dtype=float),
            np.array([[8], [4]], dtype=float),
            np.array([], dtype=float)
        ]
        nn.train_optimized([[[2, 6]], [1]], learning_rate=0.5, batch_size=1, iterations=1)
        print(nn.layers_w)


class TrainMnist(unittest.TestCase):
    @unittest.skip('')
    def test_train(self):
        nn = NeuralNet([784, 100, 10], [i for i in range(10)], optimize_weights_init=True)
        nn.train(train_set, learning_rate=0.01, batch_size=100, iterations=10)

    @unittest.skip('')
    def test_acc_load(self):
        nn = NeuralNet([784, 100, 10], [i for i in range(10)], optimize_weights_init=True)
        nn.deserialize('neural_net2020-11-15_21_00_00_483879.pickle')
        print(nn.test_accuracy(test_set))


if __name__ == "__main__":
    unittest.main()
