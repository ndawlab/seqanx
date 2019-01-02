import numpy as np
from sisyphus import ValueIteration
from sisyphus.envs.base import GraphWorld
from sisyphus.tests.common import test_world

def test_value_iteration():
    "Test value iteration dynamic programming algorithm."
    
    ## Generate test gym.
    gym = GraphWorld(*test_world())
    
    ## Test best-case learning.
    qvi = ValueIteration(policy='max', gamma=0.9)
    qvi = qvi.fit(gym)
    assert np.array_equal(qvi.Q, [ 0.9,  1. , -1. ,  0. ,  0. ])
    assert np.array_equal(qvi.V, [ 0.9,  1. ,  0. ,  0. ])
    assert np.array_equal(qvi.pi, np.arange(3))
    
    ## Test worst-case learning.
    qvi = ValueIteration(policy='min', gamma=0.9)
    qvi = qvi.fit(gym)
    assert np.array_equal(qvi.Q, [-0.9,  1. , -1. ,  0. ,  0. ])
    assert np.array_equal(qvi.V, [-0.9,  1. ,  0. ,  0. ])
    assert np.array_equal(qvi.pi, np.arange(3))
    
    ## Test betamax learning.
    qvi = ValueIteration(policy='betamax', gamma=0.9, beta=0.5)
    qvi = qvi.fit(gym)
    assert np.array_equal(qvi.Q, [ 0.0,  1. , -1. ,  0. ,  0. ])
    assert np.array_equal(qvi.V, [ 0.0,  1. ,  0. ,  0. ])
    assert np.array_equal(qvi.pi, np.arange(3))