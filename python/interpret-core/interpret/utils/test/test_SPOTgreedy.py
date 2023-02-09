import numpy as np;
from numpy import random;
from ...utils import SPOT_GreedySubsetSelection;


n = 1000; # source samples
m = 5000; # target samples
k = 10; # no. of prototypes

C = random.randint(100, size=(n, m)); # random cost matrix of size n (source) by m (target)
targetmarginal = np.ones(C.shape[1])/C.shape[1]; # uniform target distribution

prototypeIndices, prototypeWeights = SPOT_GreedySubsetSelection(C, targetmarginal, k);
print("Prototype indices: " , prototypeIndices);
print("Prototype weights: " , prototypeWeights);

assert prototypeIndices.shape[0] == k, 'The number of prototypes chosen is different from argument provided'
print("Number of chosen prototypes: " , prototypeIndices.shape[0]);
assert prototypeIndices.shape[0] == prototypeWeights.shape[0], 'The prototypes and weights vectors are of different size'

assert np.abs(np.sum(prototypeWeights) - 1)  < 1e-10, 'There is an issue with prototypes weights'
print('prototypeWeights corresponds to the weights for each prototype and its sum should be 1. Its sum is ', np.sum(prototypeWeights) )

