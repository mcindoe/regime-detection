import math
import numpy as np


NUMERIC_TYPES = (np.int, np.float, int, float)


def is_numeric(x):
    return isinstance(x, NUMERIC_TYPES)


def is_numpy(x):
    return type(x).__module__ == np.__name__


def euclid_squared(a, b):
    '''Square of euclidean distance between two points in R^n, n >= 1'''
    # If a, b are scalars
    if is_numeric(a) and is_numeric(b):
        difference = a - b
        distance = difference * difference

    # Else a, b are lists / numpy arrays
    else:
        if len(a) != len(b):
            raise ValueError('Input lengths not equal in euclid_squared()')

        if is_numpy(a) and is_numpy(b):
            difference = a-b
            distance = sum(difference * difference)
        elif (not is_numpy(a)) and (not is_numpy(b)):
            distance = 0
            for a_el, b_el in zip(a, b):
                difference = a_el - b_el
                distance += difference * difference

    return distance


def euclid_distance(a, b):
    '''Euclidean distance between two points in R^n, n >= 1'''
    squared_distance = euclid_squared(a, b)
    return math.sqrt(squared_distance)


def average_euclidean_distance(a, b):
    '''
    Returns the average Euclidean distance between elements of a
    and elements of b

    Args:
    a, b (list of lists of numerics):
        represent collections of elements of R^n for some n
        need not be the same number of elements of R^n in a and b
    '''
    
    total = 0
    for a_el in a:
        for b_el in b:
            total += euclid_distance(a_el, b_el)

    n_combinations = len(a) * len(b)
    return total / n_combinations


### BEGINNING DEFINITION OF MMD

def sum_kernels(kernel, collection, kernel_repeated_arg_value=None):
    '''
    Computes the sum of kernel(x_i, x_j) for all x_i, x_j in the
    collection, attempting to avoid some redundancy.

    Args:
    kernel: function taking two points, returning a real number
    collection: collection of points
    kernel_repeated_arg_value: if the kernel of a point with itself is
        always a particular value, this is passed instead of None and
        the computation is avoided
    '''

    assert len(collection) > 0, 'Require a non-empty collection in sum_kernels()'

    ret = 0
    
    # Compute the 'off-diagonal' sums. I.e. sum of k(x_i, x_j) for all j > i.
    for i in range(len(collection)):
        for j in range(i+1, len(collection)):
            ret += kernel(collection[i], collection[j])

    # Double to add the sum of all k(x_i, x_j) for all j < i
    ret *= 2

    # Add elements on the diagonal to get all contributions
    if kernel_repeated_arg_value is None:
        ret += sum([kernel(x, x) for x in collection])
    else:
        ret += kernel_repeated_arg_value * len(collection)

    return ret


def make_mmd_metric(kernel, kernel_repeated_arg_value=None):
    def f(X, Y):
        sum_1 = sum_kernels(kernel, X, kernel_repeated_arg_value)
        sum_3 = sum_kernels(kernel, Y, kernel_repeated_arg_value)

        sum_2 = 0
        for x_i in X:
            for y_i in Y:
                sum_2 += kernel(x_i, y_i)

        sum_1 /= (len(X) * len(X))
        sum_3 /= (len(Y) * len(Y))
        sum_2 *= (-2 / (len(X) * len(Y)))

        total = sum_1 + sum_2 + sum_3
        if np.isclose(total, 0):
            return 0
        return math.sqrt(total)

    return f

