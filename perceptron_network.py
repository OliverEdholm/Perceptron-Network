'''
Percepton network

A perceptron network for educational purposes.

Oliver Edholm 2016-10-24 13:20
'''
# imports
import numpy as np

from six.moves import xrange
from six.moves import cPickle as pickle


# functions
def perceptron(sub_x, W):
    return int(np.sum(sub_x * W) >= 0)


def insert_bias(inp):
    new_inp = list(inp)  # copy of inp
    new_inp.insert(0, 1)

    return new_inp


def train_perceptron_network(inputs, answers, max_iterations=10000):
    X = [np.array(insert_bias(inp)) for inp in inputs]
    y = answers
    W = np.zeros(len(inputs[0]) + 1)

    iterations = 0
    all_correct = False
    while not all_correct:
        if iterations == max_iterations:
            break
        all_correct = True

        for sub_x, sub_y in zip(X, y):
            y_hat = perceptron(sub_x, W)  # y_hat sounds fancier than answer

            if y_hat != sub_y:
                all_correct = False

                if y_hat == 0:
                    W += sub_x
                elif y_hat == 1:
                    W -= sub_x
        iterations += 1

    return W, iterations, all_correct


def test_input(W, inp, answer):
    inp = np.array(insert_bias(inp))

    return perceptron(inp, W) == answer


def main():
    # put data here, inputs can have higher and lower dimensions than two
    save_weights = True
    # input is 1 if the first value in the input is 1 or greater
    inputs = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 10], [2, 10]]
    answers = [0, 1, 0, 1, 0, 1, 0]

    # training
    weights, n_iterations, is_flawless = train_perceptron_network(inputs[1:],
                                                                  answers[1:])

    # "evaluate", you should use more data, than this
    if test_input(weights, inputs[0], answers[0]):
        print('Perceptron network is working!')
    else:
        print('Perceptron network is not working!')

    # saving
    if save_weights:
        print('Saving weights...')
        with open('perceptron_weights.pkl', 'wb') as pkl_file:
            pickle.dump(weights, pkl_file)


if __name__ == '__main__':
    main()

