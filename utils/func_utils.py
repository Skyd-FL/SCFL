import random
import math


def uniform_generator(mean, std):
    upper_b = mean+std
    lower_b = mean-std
    return random.uniform(lower_b, upper_b)
