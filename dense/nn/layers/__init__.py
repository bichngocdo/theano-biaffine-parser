import math


def initialize_weight_matrix(no_input, no_output):
    return np.random.randn(no_input, no_output) / math.sqrt(no_input)


def initialize_bias(size):
    return np.ones(size) * 0.01


from core import *
from transform import *
from merge import *
