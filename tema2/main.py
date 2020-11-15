import gzip
import pickle
import rnlib.perceptron_net as pn
import rnlib.perceptron as p
import numpy as np
import unittest


class TestPerceptrons(unittest.TestCase):
    def test_over_standard(self):
        with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin')

            net = pn.SingleLayerNet([x for x in range(10)])
            # net.train(train_set[0], train_set[1], batches=10) this is actually worse
            net.train_no_batch(train_set[0], train_set[1])
            ac1 = net.test_accuracy(valid_set[0], valid_set[1])
            ac2 = net.test_accuracy(test_set[0], test_set[1])
            self.assertGreater(ac1, 0.80)
            self.assertGreater(ac2, 0.80)

    def test_parallel_serial_same(self):
        """
        Tests that training perceptrons in serial and in parallel provides the same result
        """
        with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin')

            net = pn.SingleLayerNet([x for x in range(10)])
            # net.train(train_set[0], train_set[1], batches=10)
            net.train_no_batch(train_set[0], train_set[1])
            # net.show()
            ac1 = net.test_accuracy(valid_set[0], valid_set[1])
            ac2 = net.test_accuracy(test_set[0], test_set[1])

            ww = [0 for x in range(10)]
            bb = [0 for z in range(10)]
            for t in range(10):
                ww[t], bb[t] = p.adeline_perceptron(zip(train_set[0], [t == ti for ti in train_set[1]]), len(train_set[0][0]))
            ac3 = 0
            for x, t in zip(valid_set[0], valid_set[1]):
                res = [p.compute_result_for_perceptron(ww[i], bb[i], x) for i in range(10)]
                ac3 += (t == np.argmax(res))
            ac3 = ac3 / len(valid_set[0])
            ac4 = 0
            for x, t in zip(test_set[0], test_set[1]):
                res = [p.compute_result_for_perceptron(ww[i], bb[i], x) for i in range(10)]
                ac4 += (t == np.argmax(res))
            ac4 = ac4 / len(test_set[0])

            self.assertEqual(ac1, ac3)
            self.assertEqual(ac2, ac4)


if __name__ == "__main__":
    with gzip.open('../files/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin')

        net = pn.SingleLayerNet([x for x in range(10)])
        # net.train(train_set[0], train_set[1], batches=10) this is actually worse
        net.train(train_set[0], train_set[1], batches=10)
        ac1 = net.test_accuracy(valid_set[0], valid_set[1])
        ac2 = net.test_accuracy(test_set[0], test_set[1])
        print(ac1, ac2 )
    # unittest.main()
