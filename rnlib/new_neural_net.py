from datetime import datetime

import numpy as np
import pickle


class NewNeuralNet:
    def __init__(self, layers, learning_rate, dropout_p):
        self.num_layers = len(layers)
        self.layer_sizes = layers
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p
        # variance
        sigma = [1 / np.sqrt(layers[i]) for i in range(self.num_layers - 1)]

        self.biases = [np.random.normal(scale=sigma[i], size=(layers[i + 1], 1)) for i in range(self.num_layers - 1)]
        self.weights = [np.random.normal(scale=sigma[i], size=(layers[i + 1], layers[i])) for i in range(self.num_layers - 1)]

    def train(self, train_set, iterations, eval_data):
        train_data, train_labels = train_set

        for j in range(iterations):
            print('it')
            seed = np.random.randint(0, 100000)
            np.random.seed(seed)
            np.random.shuffle(train_data)
            np.random.seed(seed)
            np.random.shuffle(train_labels)

            batches = []

            for i in range(0, len(train_data), 10):
                batches.append((train_data[i:i + 10], train_labels[i:i + 10]))

            for batch in batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                dropout = np.array([np.random.choice(2, self.layer_sizes[1], p=[self.dropout_p, 1 - self.dropout_p])]).transpose()

                for x, y in zip(batch[0], batch[1]):
                    delta_nabla_b, delta_nabla_w = self.backprop(x, y, dropout)

                    nabla_b = [nabla_b[i] + delta_nabla_b[i] for i in range(len(nabla_b))]
                    nabla_w = [nabla_w[i] + delta_nabla_w[i] for i in range(len(nabla_w))]

                self.weights = [self.weights[i] - (nabla_w[i] * self.learning_rate) for i in range(len(self.weights))]
                self.biases = [self.biases[i] - (nabla_b[i] * self.learning_rate) for i in range(len(self.biases))]

                break

            accuracy = self.test_accuracy(eval_data)
            print(accuracy/len(eval_data[0]))

        self.serialize('network.pickle')

    def backprop(self, x, y, dropout):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        z_values, activations = self.feed_forward(x, dropout)

        y = np.array([[1] if y == i else [0] for i in range(10)])

        delta = (activations[-1] - y) * sigmoid_prime(activations[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = z_values[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def feed_forward(self, x, dropout=None, testing=False):
        x = np.array([x]).transpose()
        zs = []
        xs = [x]

        for i in range(0, self.num_layers - 1):
            z = np.dot(self.weights[i], x) + self.biases[i]
            zs.append(z)
            if i != self.num_layers - 2:
                if not testing:
                    z = dropout * z / (1 - self.dropout_p)
                x = sigmoid(z)
            else:
                x = softmax(z)
            xs.append(x)
        return zs, xs

    def test_accuracy(self, eval_data):
        data = eval_data[0]
        labels = eval_data[1]

        results = [(np.argmax(self.feed_forward(x, testing=True)[1][-1]), y) for (x, y) in zip(data, labels)]
        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def serialize(self, name=None):
        if name is None:
            name = 'neural_net' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_') + '.pickle'
        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            self.learning_rate = obj.learning_rate
            self.dropout_p = obj.dropout_p
            self.num_layers = obj.num_layers
            self.layer_sizes = obj.layer_sizes
            self.biases = obj.biases
            self.weights = obj.weights


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z)))
