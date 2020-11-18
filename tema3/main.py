import gzip
import os
import pickle
import numpy as np
from neural_net import NeuralNet
from neural_net_activations import softmax
from new_neural_net import NewNeuralNet

with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

if __name__ == "__main__":
    nn = NeuralNet([784, 100, 10], [i for i in range(10)], last_activation=softmax, optimize_weights_init=True)
    nn.train_optimized(train_set, learning_rate=0.1, batch_size=100, iterations=200, test_data=valid_set, save=True)
    print(nn.test_accuracy(valid_set))

    # net = NewNeuralNet([784, 100, 10], learning_rate=0.01, dropout_p=0.2)
    # net.train(train_set=train_set, iterations=30, eval_data=test_set)
    # print(net.test_accuracy(valid_set))

    # net = NeuralNet([1])
    # for root, directories, files in os.walk('.'):
    #     files = [os.path.join(root, f) for f in files if f.startswith('neural_net')]
    #     for f in files[2:]:
    #         try:
    #             net.deserialize(f)
    #             print(f, net.test_accuracy(test_set))
    #         except Exception as e:
    #             pass

