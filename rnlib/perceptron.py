import numpy as np


MAX_ITERATIONS = 10


def perceptron_activation(ii: int):
    return 1 if ii > 0 else 0


def perceptron_online_training(training_set, num_of_features):
    weights = np.array([0 for x in range(num_of_features)])
    all_classified = False
    nr_iterations = MAX_ITERATIONS
    b = 0
    miu = 0.01

    while not all_classified and nr_iterations > 0:
        all_classified = True
        for i, (x, t) in enumerate(training_set):
            z = np.dot(weights, x) + b
            output = perceptron_activation(z)
            weights += (t - output) * x * miu
            b += (t - output) * miu
            if output != t:
                all_classified = False
        nr_iterations -= 1
    return weights, b


def perceptron_batch_training(training_set, num_of_features):
    weights = np.array([0 for x in range(num_of_features)])
    nr_iterations = MAX_ITERATIONS
    b = 0
    miu = 0.01
    batches = len(training_set) // 100

    while nr_iterations > 0:
        alfas = np.array([0 for x in range(batches)])
        betas = np.array([0 for x in range(batches)])
        for ba in range(batches):
            alfas[ba] = np.array([0 for x in range(num_of_features)])
            betas[ba] = 0
            for v, (x, t) in enumerate(training_set[ba * len(training_set) / batches:(ba + 1) * len(training_set) / batches]):
                z = np.dot(weights, x) + b
                output = perceptron_activation(z)
                alfas[ba] += (t - output) * x * miu
                betas[ba] += (t - output) * miu

        for ba in range(batches):
            weights += alfas[ba]
            b += betas[ba]
        nr_iterations -= 1
    return weights, b


def perceptron_mini_batch_training(training_set, num_of_features):
    weights = np.array([0 for x in range(num_of_features)])
    nr_iterations = MAX_ITERATIONS
    b = 0
    miu = 0.01
    batches = len(training_set) // 100

    while nr_iterations > 0:
        for ba in range(batches):
            alfa = np.array([0 for x in range(num_of_features)])
            beta = 0
            for v, (x, t) in enumerate(training_set[ba * len(training_set) / batches:(ba + 1) * len(training_set) / batches]):
                z = np.dot(weights, x) + b
                output = perceptron_activation(z)
                alfa += (t - output) * x * miu
                beta += (t - output) * miu
            weights += alfa
            b += beta
        nr_iterations -= 1
    return weights, b


def adeline_perceptron(training_set, num_of_features):
    weights = np.array([0.0 for x in range(num_of_features)])
    nr_iterations = MAX_ITERATIONS
    b = 0
    miu = 0.001

    while nr_iterations > 0:
        for x, t in training_set:
            z = np.dot(weights, x) + b
            weights += (float(t) - z) * x * miu
            b += (float(t) - z) * miu
        nr_iterations -= 1
    return weights, b


def compute_result_for_perceptron(weights, b, input_f):
    return np.dot(weights, input_f) + b
