import numpy as np
from scipy.stats import entropy, wasserstein_distance

def kl_divergence(p, q, eps=1e-7):
    p = p + eps
    q = q + eps
    return entropy(p, q)

def wasserstein(p, q):
    return wasserstein_distance(p, q)
