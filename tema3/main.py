import _pickle as cPickle
import gzip

from neural_net import NeuralNet
from neural_net_activations import softmax


def main():
    with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f, encoding="bytes")
    nn = NeuralNet([784, 100, 10], [i for i in range(10)], last_activation=softmax, optimize_weights_init=True)
    nn.train_optimized(train_set, learning_rate=0.01, batch_size=100, iterations=200, test_data=valid_set, save=True)
    print(nn.test_accuracy(valid_set))


if __name__ == "__main__":
    main()
