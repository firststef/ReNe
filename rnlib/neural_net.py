import pickle
import numpy as np
from datetime import datetime

from neural_net_activations import sigmoid
from neural_net_cost import CrossEntropyCost


class NeuralNet:
    def __init__(
            self,
            layers_dimensions=None,
            out_classes=None,
            cost=CrossEntropyCost(),
            activation=sigmoid,
            last_activation=sigmoid,
            dont_update_biases=False,
            optimize_weights_init=False
    ):
        if out_classes is None:
            out_classes = []
        if layers_dimensions is None:
            layers_dimensions = []
        self.layers_dimensions = layers_dimensions
        self.layers_num = len(layers_dimensions)
        self.out_classes = out_classes
        self.layers_w = None
        self.layers_b = None
        self.prev_ys = None
        self.cost = cost
        self.activation = activation
        self.last_activation = last_activation
        self.dont_update_weights = dont_update_biases
        self.optimize_weights_init = optimize_weights_init
        self.dropout_p = 0.2
        self.init()

    def init(self):
        self.layers_w = [np.array([0], dtype=float) for i in range(self.layers_num)]
        self.layers_b = [np.array([0], dtype=float) for i in range(self.layers_num)]
        self.prev_ys = [np.array([0], dtype=float) for i in range(self.layers_num)]

        mean, std_dev = 0, 1 / self.layers_dimensions[0] ** 0.5

        for i, l in enumerate(self.layers_dimensions):
            if not self.optimize_weights_init:
                self.layers_w[i] = np.full(l, 0, dtype=float)
            else:
                self.layers_w[i] = np.random.normal(mean, std_dev, l)

            self.prev_ys[i] = np.full(l, 0, dtype=float)
            if i == self.layers_num - 1:
                break

            l2 = self.layers_dimensions[i + 1]
            if not self.optimize_weights_init:
                self.layers_w[i] = np.full((l, l2), 0, dtype=float)
                self.layers_b[i + 1] = np.full(l2, 0, dtype=float)
            else:
                self.layers_w[i] = np.random.normal(mean, std_dev, size=(l, l2))
                self.layers_b[i + 1] = np.random.normal(mean, std_dev, size=l2)

    def train(self, train_data, learning_rate, batch_size, iterations, save=True, test_data=None):
        # Inputs
        inputs = train_data[0]
        values = train_data[1]
        num_of_tests = len(inputs)

        # Aliasing
        layers_num = self.layers_num
        layers_w = self.layers_w
        layers_b = self.layers_b
        batches = num_of_tests // batch_size

        # Init deltas
        delta_w = [np.array([0], dtype=float) for i in range(self.layers_num)]
        delta_b = [np.array([0], dtype=float) for i in range(self.layers_num)]
        for i, l in enumerate(self.layers_dimensions):
            delta_w[i] = np.full(l, 0, dtype=float)
            if i == self.layers_num - 1:
                break
            l2 = self.layers_dimensions[i + 1]
            delta_w[i] = np.full((l, l2), 0, dtype=float)
            delta_b[i + 1] = np.full(l2, 0, dtype=float)

        for it in range(iterations):
            seed = np.random.randint(0, 100000)
            np.random.seed(seed)
            np.random.shuffle(inputs)
            np.random.seed(seed)
            np.random.shuffle(values)

            for ba in range(batches):
                # mini-batch setup
                for wl in delta_w:
                    wl.fill(0)
                for wb in delta_b:
                    wb.fill(0)

                for i in range(batch_size):
                    x = inputs[ba * batch_size + i]
                    t = values[ba * batch_size + i]

                    # feed forward
                    self.prev_ys[0] = np.array(x, dtype=float)
                    inp = x
                    print("FORWARD")
                    for ll in range(self.layers_num - 1):
                        print('strat ' + str(ll+1) + ':')
                        inp = self.feed_forward(ll, inp)
                        self.prev_ys[ll + 1] = inp

                    # compute error for the last layer output
                    # note: this error could instead be evaluated for the entire network
                    ts = [(1 if v == t else 0) for v in self.out_classes]
                    err = self.cost.error(inp, ts)

                    # back propagation
                    print("BACKPROPAGATION")
                    errs = np.array(err, dtype=float)
                    for l in range(self.layers_num - 1, 0, -1):
                        # print("eroare δ stratul " + str(l+1) + ":", errs)
                        [print("δ({}){}: ".format(l+1, i+1) + str(_err)) for i, _err in enumerate(errs)]
                        delta_batch_w, delta_batch_b = self.back_propagate(l, errs, learning_rate)
                        delta_w[l - 1] += delta_batch_w
                        delta_b[l] += delta_batch_b
                        errs = self.compute_layer_error(l - 1, errs)

                layers_b = [layers_b[vi] + delta_b[vi] for vi in range(layers_num)]
                layers_w = [layers_w[vi] + delta_w[vi] for vi in range(layers_num)]

            print("END")
            self.layers_w = layers_w
            self.layers_b = layers_b

            if test_data:
                print(self.test_accuracy(test_data))

        if save:
            self.serialize()

    def train_optimized(self, train_data, learning_rate, batch_size, iterations, save=True, test_data=None):
        name = 'neural_net' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_') + '.pickle'

        # self.deserialize('neural_net2020-11-17_09_55_59_645857.pickle')

        # Aliasing
        layers_num = self.layers_num
        layers_w = self.layers_w
        layers_b = self.layers_b
        prev_ys = self.prev_ys
        activation = self.activation
        last_activation = self.last_activation
        out_classes = self.out_classes
        layers_dimensions = self.layers_dimensions
        cost = self.cost

        eps = 0.00000001
        gamma = 0.9

        # Inputs
        inputs = train_data[0]
        values = train_data[1]
        num_of_tests = len(inputs)
        batches = num_of_tests // batch_size

        if self.dont_update_weights:
            print('[ WARN ]: Dont update weights only works in train()')

        # Init deltas
        delta_w = [np.array([0], dtype=float) for i in range(self.layers_num)]
        delta_b = [np.array([0], dtype=float) for i in range(self.layers_num)]
        for i, l in enumerate(self.layers_dimensions):
            delta_w[i] = np.full(l, 0, dtype=float)
            if i == self.layers_num - 1:
                break
            l2 = self.layers_dimensions[i + 1]
            delta_w[i] = np.full((l, l2), 0, dtype=float)
            delta_b[i + 1] = np.full(l2, 0, dtype=float)

        for it in range(iterations):
            # seed = np.random.randint(0, 100000)
            # np.random.seed(seed)
            # np.random.shuffle(inputs)
            # np.random.seed(seed)
            # np.random.shuffle(values)

            for ba in range(batches):
                # print('batch ' + str(ba))
                # mini-batch setup
                for wl in delta_w:
                    wl.fill(0)
                for wb in delta_b:
                    wb.fill(0)

                self.dropout_p = 0.2

                for i in range(batch_size):
                    x = inputs[ba * batch_size + i]
                    t = values[ba * batch_size + i]

                    # dropout = np.array([np.random.choice(2, 100, p=[self.dropout_p, 1 - self.dropout_p])]).transpose()

                    # feed forward
                    prev_ys[0] = np.array(x, dtype=float)
                    inp = x
                    for la in range(layers_num - 1):
                        if la == layers_num - 1:
                            z = np.dot(inp, layers_w[la]) + layers_b[la + 1]
                            # z = np.array([dropout[di] * z[di] for di in range(len(z))], dtype=float) / (1 - self.dropout_p)
                            inp = last_activation(z)
                        else:
                            z = np.dot(inp, layers_w[la]) + layers_b[la + 1]
                            # for di in range(len(z)):
                            #     z[di] *= dropout[di] / (1 - self.dropout_p)
                            inp = activation(z)
                        prev_ys[la + 1] = inp

                    # compute error for the last layer output
                    # note: this error could instead be evaluated for the entire network
                    ts = np.array([1 if v == t else 0 for v in out_classes], dtype=float)
                    err = cost.error(inp, ts)

                    # back propagation
                    errs = np.array(err, dtype=float)
                    for l in range(layers_num - 1, 0, -1):
                        # apply RMSProp
                        # ln = learning_rate / np.sqrt(gamma * np.power(sum(errs), 2) + (1 - gamma) * np.power(sum(err), 2) + eps)
                        ln = learning_rate

                        delta_b[l] -= np.multiply(ln, errs)
                        for rp in range(layers_dimensions[l - 1]):
                            delta_w[l - 1][rp] -= np.multiply(errs, ln * prev_ys[l - 1][rp])
                            # a = np.multiply(errs, ln * prev_ys[l - 1][rp])
                            # ln = learning_rate / np.sqrt(
                            #     gamma * np.power(a, 2) + (1 - gamma) * np.power(a, 2) + eps)
                        errs = prev_ys[l - 1] * (1 - prev_ys[l - 1]) * np.dot(layers_w[l - 1], errs)

                layers_b = [layers_b[vi] + delta_b[vi] for vi in range(layers_num)]
                layers_w = [layers_w[vi] + delta_w[vi] for vi in range(layers_num)]

            self.layers_w = layers_w
            self.layers_b = layers_b

            if save:
                self.serialize(name)

            if test_data:
                print(self.test_accuracy(test_data))

    def feed_forward(self, layer, inputs):
        if layer == self.layers_num - 1:
            return None

        z = np.dot(inputs, self.layers_w[layer]) + self.layers_b[layer + 1]
        print("z-uri:", z)
        ys = self.activation(z)
        print("y-uri:", ys)

        return ys

    def compute_layer_error(self, layer, errors):
        return [self.prev_ys[layer][i] * (1 - self.prev_ys[layer][i]) * (errors * self.layers_w[layer][i])[0] for i in range(self.layers_dimensions[layer])]

    def back_propagate(self, layer, errs, learning_rate):
        delta_b = np.full_like(self.layers_b[layer], 0)
        delta_w = np.full_like(self.layers_w[layer - 1], 0)
        if not self.dont_update_weights:
            delta_b -= np.multiply(learning_rate, errs)
        for i in range(self.layers_dimensions[layer - 1]):
            [print("derivata δC/δw({}){}{}: {}".format(layer+1, i+1, j+1, str(_dc))) for j, _dc in enumerate(np.multiply(errs, self.prev_ys[layer - 1][i]))]
            delta_w[i] -= np.multiply(errs, learning_rate * self.prev_ys[layer - 1][i])
        return delta_w, delta_b

    def serialize(self, name=None):
        if name is None:
            name = 'neural_net' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_') + '.pickle'
        with open(name, 'wb') as f:
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
            self.layers_num = len(self.layers_dimensions)

    def test_one(self, input_data):
        inp = input_data
        for l in range(self.layers_num - 1):
            inp = self.feed_forward(l, inp)
        return self.out_classes[np.argmax(inp)]

    def test_accuracy(self, test_set):
        input_data = test_set[0]
        outputs = test_set[1]
        num_valid = 0
        for di, t in zip(input_data, outputs):
            num_valid += (self.test_one(di) == t)
        return num_valid / len(input_data)
