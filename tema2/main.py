import gzip
import pickle
import rnlib.perceptron as perceptron

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')

    zero_set = [(x, 0 if t != 0 else 1) for x, t in zip(train_set[0], train_set[1])]
    weights, b = perceptron.adeline_perceptron(zero_set, len(train_set[0][0]))

    print(perceptron.compute_result_for_perceptron(weights, b, test_set[0][0]))
    print(perceptron.compute_result_for_perceptron(weights, b, test_set[0][1]))
