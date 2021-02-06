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
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        nn = NeuralNet([2, 3, 1], [0], last_activation=softmax, dont_update_biases=True)
        nn.layers_w = [
            np.array([[0, -2, -4], [4, 1, 2]], dtype=float),
            np.array([[2], [4], [6]], dtype=float),
            np.array([], dtype=float)
        ]
        nn.train([[[1, 2]], [1]], learning_rate=0.5, batch_size=1, iterations=1)
        print(nn.layers_w)


class TrainMnist(unittest.TestCase):
    @unittest.skip('')
    def test_train(self):
        nn = NeuralNet([784, 100, 10], [i for i in range(10)], last_activation=softmax, optimize_weights_init=False)
        nn.train_optimized(train_set, learning_rate=0.01, batch_size=100, iterations=200, test_data=valid_set, save=True)
        print(nn.test_accuracy(valid_set))

    @unittest.skip('')
    def test_acc_load(self):
        nn = NeuralNet([784, 100, 10], [i for i in range(10)], optimize_weights_init=True)
        nn.deserialize('neural_net2020-11-19_14_43_51_840922.pickle')
        print(nn.test_accuracy(test_set))


if __name__ == "__main__":
    unittest.main()
