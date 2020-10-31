import gzip
import pickle
import rnlib.perceptron_net as pn

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

    net = pn.SingleLayerNet([x for x in range(10)])
    net.train(train_set[0], train_set[1])
    # net.show()
    print(net.test_accuracy(test_set[0], test_set[1]))
