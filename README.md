# ReNe-
My laboratory work over the course of the first semester of year 2020-2021 for Neural Networks - UAIC Iasi

Tema 1:
Creati doua scripturi in python care rezolva un sistem liniar de ecuatii cu 3 cunoscute. Primul script rezolva sistemul fara a folosi libraria numpy (doar cu structurile de date si functiile standard din python), iar al doilea script repeta pasii din primul script, dar folosind functii si structuri din numpy. Verificati ca cele doua scripturi dau acelasi rezultat.

Datele de intrare vor fi date intr-un fisier pe 3 linii de forma:
  a1x + b1y + c1z = r1
  a2x + b2y + c2z = r2
  a3x + b3y + c3z = r3
  
Tema 2:
Implement a model, based on the perceptron algorithm that will be able to classify hand written digits
For training, use the mnist dataset. This can be found, in a format that can be easily worked with in
python, at the following url: http://deeplearning.net/data/mnist/mnist.pkl.gz
The dataset is split in 3 sets: training_set, validation_set, test_set. Each of these 3 sets contains two
vectors of equal length:
1. A set of digits written as a vector of length 784. The digits from the mnist dataset have the
shape 28x28 pixels and are represented as a vector ( each of the 28 lines from the 28x28 matrix
are written one after each other, thus forming a vector of 784 elements). Each pixel from the
matrix has a value between 0 and 1, where 0 represents white, 1 represents black and the value
between 0 and 1 is a shade of grey.
2. A label for each element from the first vector: a number between 0 and 9 representing the digit
from the image

The 3 sets have the following meaning:
- training_set (used for training your model); 50000 elements
- validation_set (usually used to adjust hyper-parameters and to perform a first evaluation of the
resulted model); 10000 elements
- test_set (dataset used for testing. Use it only after you’ve fine-tuned the algorithm using the
validation set. Do not use it for fine-tuning); 10000 elements

The classification algorithm must be based on 10 perceptrons. Each of these 10 perceptron will be
trained to classify images that represent only one digit. For example, the first perceptron will be trained
to output value of 1 for the digit 0 and the value 0 for every other digit).
When each perceptron has been successfully trained and you want to see how the classification works,
the input will be fed to each perceptron and the class will be given by the perceptron which outputs a 1
or the perceptron who has the biggest net input
