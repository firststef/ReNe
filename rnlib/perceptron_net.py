from math import sqrt
import numpy as np
import pickle
import matplotlib.pyplot as plt

MAX_ITERATIONS = 10


class SingleLayerNet:
    def __init__(self, out_classes):
        self.out_classes = out_classes
        self.perceptrons_w = None
        self.perceptrons_b = None

    def train_no_batch(self, inputs, values):
        nr_iterations = MAX_ITERATIONS
        learning_rate = 0.001
        num_of_edges = len(inputs[0])
        num_of_classes = len(self.out_classes)
        perceptrons_w = np.array(np.full((num_of_classes, num_of_edges), 0), dtype=np.float)
        perceptrons_b = np.array(np.full(num_of_classes, 0), dtype=np.float)
        t_set = zip(inputs, values)

        while nr_iterations > 0:
            # adaline algorithm
            for x, t in t_set:
                z_m = np.dot(perceptrons_w, x) + perceptrons_b
                t_m = np.array([1 if t == cl else 0 for cl in self.out_classes])
                # a_m = self.activation_arr(z_m)
                perceptrons_w += np.array([ou * x * learning_rate for ou in np.array(t_m - z_m)])
                # np.outer [10, 1] (t_m - z_m) * reshape x => [1, 700] * learning_rate
                perceptrons_b += (t_m - z_m) * learning_rate
            nr_iterations -= 1
        self.perceptrons_w = perceptrons_w
        self.perceptrons_b = perceptrons_b

        with open('percep.pickle', 'wb') as f:
            pickle.dump([perceptrons_w, perceptrons_b], f)

    def train(self, inputs, values, batches=None):
        nr_iterations = MAX_ITERATIONS
        learning_rate = 0.001
        num_of_edges = len(inputs[0])
        num_of_classes = len(self.out_classes)
        num_of_tests = len(inputs)
        perceptrons_w = np.array(np.full((num_of_classes, num_of_edges), 0), dtype=np.float)
        perceptrons_b = np.array(np.full(num_of_classes, 0), dtype=np.float)
        if not batches:
            batches = num_of_tests // 10
        ba_size = num_of_tests // batches
        batches_z = [zip(inputs[ba * ba_size:(ba + 1) * ba_size], values[ba * ba_size:(ba + 1) * ba_size]) for ba in range(batches)]

        alfas = np.array(np.full((num_of_classes, num_of_edges), 0), dtype=np.float)
        beta = np.array(np.full(num_of_classes, 0), dtype=np.float)
        while nr_iterations > 0:
            # adaline algorithm
            for ba in range(batches):
                # mini-batch setup
                alfas.fill(0)
                beta.fill(0)

                for x, t in batches_z[ba]:  # zip only cycles once through its pairs, change this to indexes
                    z_m = np.dot(perceptrons_w, x) + perceptrons_b
                    t_m = np.array([1 if t == cl else 0 for cl in self.out_classes])
                    # a_m = self.activation_arr(z_m)
                    alfas += np.array([ou * x * learning_rate for ou in np.array(t_m - z_m)])
                    beta += (t_m - z_m) * learning_rate

            perceptrons_w += alfas / ba_size
            perceptrons_b += beta / ba_size
            nr_iterations -= 1
        self.perceptrons_w = perceptrons_w
        self.perceptrons_b = perceptrons_b

        with open('percep.pickle', 'wb') as f:
            pickle.dump([perceptrons_w, perceptrons_b], f)

    def activation(self, inp):
        return 1 if inp > 0 else -1

    def activation_arr(self, arr):
        return [self.activation(x) for x in arr]

    def test_one(self, input_data):
        z_m = np.dot(self.perceptrons_w, input_data) + self.perceptrons_b
        return self.out_classes[np.argmax(z_m)]

    def test_accuracy(self, input_data, outputs):
        num_valid = 0
        for di, t in zip(input_data, outputs):
            num_valid += (self.test_one(di) == t)
        return num_valid / len(input_data)

    def show(self):
        for i, w in enumerate(self.perceptrons_w):
            data = np.reshape((w + 1) / 2, [28, 28])
            data = ((data - data.min()) / (data.max() - data.min())) * 255
            plt.title(str(self.out_classes[i]))
            plt.imshow(data.astype(np.uint8))
            plt.show()