import numpy as np
from interpret.utils import SPOT_GreedySubsetSelection
from numpy import random

n = 1000
# source samples
m = 5000
# target samples
k = 10
# no. of prototypes

C = random.randint(100, size=(n, m))
# random cost matrix of size n (source) by m (target)
targetmarginal = np.ones(C.shape[1]) / C.shape[1]
# uniform target distribution

prototypeIndices, prototypeWeights = SPOT_GreedySubsetSelection(C, targetmarginal, k)

assert prototypeIndices.shape[0] == k, (
    "The number of prototypes chosen is different from argument provided"
)
assert prototypeIndices.shape[0] == prototypeWeights.shape[0], (
    "The prototypes and weights vectors are of different size"
)

assert np.abs(np.sum(prototypeWeights) - 1) < 1e-10, (
    "There is an issue with prototypes weights"
)
