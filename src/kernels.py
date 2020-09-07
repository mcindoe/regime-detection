from math import exp
import numpy as np

from metrics import euclid_squared
from metrics import euclid_distance

def make_laplacian_kernel(sigma):
    def k(x, y):
        squared_distance = euclid_squared(x, y)
        power = -sigma*sigma*squared_distance
        return exp(power)
    return k

if False:
    def make_laplacian_kernel(sigma):
        def k(x, y):
            distance = euclid_distance(x, y)
            power = -(distance / sigma)
            return exp(power)

        return k


def gaussian_kernel(x, y):
    dot_product = np.dot(x, y)
    return exp(dot_product)
