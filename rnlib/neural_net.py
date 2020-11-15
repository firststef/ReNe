import math
from datetime import datetime

import numpy as np
import pickle

MAX_ITERATIONS = 10

# todo: test how bad is python when calling functions


class NeuralNet:
    def __init__(self, layers_dimensions=None, out_classes=None, dont_update_weights=False):
        if out_classes is None:
            out_classes = []
        if layers_dimensions is None:
            layers_dimensions = []
        self.layers_dimensions = layers_dimensions
        self.layers_num = len(layers_dimensions)
        self.out_classes = out_classes
        self.learning_rate = None
        self.layers_w = None
        self.layers_b = None
        self.prev_ys = None
        self.dont_update_weights = dont_update_weights
        # self.errors = None
        self.init()

    def init(self):
        self.layers_w = [np.array([], dtype=float) for i in range(self.layers_num)]
        self.layers_b = [np.array([], dtype=float) for i in range(self.layers_num)]
        self.prev_ys = [np.array([], dtype=float) for i in range(self.layers_num)]
        # self.errors = np.full(self.layers_num + 1, [])  # virtually adding a final result node to make error computing similar
        for i, l in enumerate(self.layers_dimensions):
            self.layers_w[i] = np.full(l, 0, dtype=float)  # de initializat cu distributia aia
            self.prev_ys[i] = np.full(l, 0, dtype=float)
            # self.errors[i] = np.full(l, 0)
            if i == self.layers_num - 1:
                break
            l2 = self.layers_dimensions[i + 1]
            self.layers_w[i] = np.full((l, l2), 0, dtype=float)  # de initializat cu distributia aia
            self.layers_b[i + 1] = np.full(l2, 0, dtype=float)  # de initializat cu distributia aia

    def train(self, inputs, values, learning_rate, batch_size, iterations):
        self.learning_rate = learning_rate
        num_of_tests = len(inputs)
        batches = num_of_tests // batch_size
        layers_w = self.layers_w
        layers_b = self.layers_b

        for it in range(iterations):
            # shuffle inputs

            for ba in range(batches):
                # mini-batch setup

                # setup mean for batch
                for i in range(batch_size):
                    x = inputs[ba * batch_size + i]
                    t = values[ba * batch_size + i]

                    # feed forward
                    self.prev_ys[0] = np.array(x, dtype=float)
                    inp = x
                    for l in range(self.layers_num - 1):
                        inp = self.feed_forward(l, inp, True)
                    # chosen_y = self.out_classes[np.argmax(inp)]
                    y = inp[t]

                    # compute error
                    err = self.compute_net_result_error(y, 1)

                    # back propagation
                    errs = np.array([err], dtype=float)
                    for l in range(self.layers_num - 1, 0, -1):
                        errs = self.back_propagate(l, errs)

            self.layers_w = layers_w
            self.layers_b = layers_b

        self.serialize()

    def feed_forward(self, layer, inputs, save=False):
        if layer == self.layers_num - 1:
            return None

        # ys = [np.array([]) for i in range(self.layers_dimensions[layer + 1])]

        z = np.dot(inputs, self.layers_w[layer]) + self.layers_b[layer + 1]
        ys = self.activation(z)

        if save and layer != self.layers_num - 1:
            if len(self.prev_ys[layer + 1]) != len(ys):
                raise BaseException("Ai cam daton bara gigele")
            self.prev_ys[layer + 1] = ys

        return ys

    @staticmethod
    def activation(v):
        """ Sigmoid activation """
        return 1 / (1 + math.e ** -v)

    @staticmethod
    def compute_net_result_error(y, t):
        return y - t

    def compute_result_layer_error(self, layer, errors):
        return [self.prev_ys[layer][i] * (1 - self.prev_ys[layer][i]) * (errors * self.layers_w[layer][i])[0] for i in range(self.layers_dimensions[layer])]

    def back_propagate(self, layer, errs):
        """
        :param layer:
        :param errs: is list
        :return: the errors for the current layer
        """
        if layer == 0:
            return None

        # l2 = 1 if layer == self.layers_num - 1 else self.layers_dimensions[layer]
        if not self.dont_update_weights:
            self.layers_b[layer] -= np.multiply(self.learning_rate, errs)
        for i in range(self.layers_dimensions[layer - 1]):
            # a = self.learning_rate
            # b = errs
            # c = self.prev_ys[layer - 1][i]
            self.layers_w[layer - 1][i] -= np.multiply(errs, self.learning_rate * self.prev_ys[layer - 1][i])
        return self.compute_result_layer_error(layer - 1, errs)

    def serialize(self):
        with open('neural_net' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_') + '.pickle', 'wb') as f:
            pickle.dump({
                'layers_b': self.layers_b,
                'layers_w': self.layers_w,
                'layers_dimensions': self.layers_dimensions
            }, f)

    def deserialize(self, file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            self.layers_b = obj["layers_b"]
            self.layers_w = obj["layers_w"]
            self.layers_dimensions = obj["layers_dimensions"]

    def test_one(self, input_data):
        inp = input_data
        for l in range(self.layers_num - 1):
            inp = self.feed_forward(l, inp)
        return self.out_classes[np.argmax(inp)]

    def test_accuracy(self, input_data, outputs):
        num_valid = 0
        for di, t in zip(input_data, outputs):
            num_valid += (self.test_one(di) == t)
        return num_valid / len(input_data)
