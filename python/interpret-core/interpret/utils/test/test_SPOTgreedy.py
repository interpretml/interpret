import numpy as np;
from numpy import random;
from ...utils import SPOT_GreedySubsetSelection;


n = 1000; # source samples
m = 5000; # target samples
k = 10; # no. of prototypes

C = random.randint(100, size=(n, m)); # random cost matrix
targetmarginal = np.ones(C.shape[1])/C.shape[1]; # uniform target marginal

prototypeIndices, prototypeWeights = SPOT_GreedySubsetSelection(C, targetmarginal, k);
print("Prototype indices -- " , prototypeIndices);
print("Prototype weights -- " , prototypeWeights);
