import numpy as np
from sisyphus.mdp import ModelFree
from sisyphus.envs._base import GraphWorld
from sisyphus.tests.common import test_world

def test_model_free():
    """Test model free temporal difference learning algorithm."""
    np.random.seed(47404)

    ## Generate test gym.
    gym = GraphWorld(*test_world())

    ## Define exploration schedule.
    schedule = np.zeros(100)

    ## Test best-case learning.
    agent = ModelFree(policy='max', eta=0.2, gamma=0.9)
    agent = agent.fit(gym, choice='softmax', schedule=schedule)
    assert np.allclose(agent.Q, [ 0.9,  1. , -1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.allclose(agent.V, [ 0.9,  1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.array_equal(agent.pi, np.arange(3))

    ## Test worst-case learning.
    agent = ModelFree(policy='min', eta=0.2, gamma=0.9)
    agent = agent.fit(gym, choice='softmax', schedule=schedule)
    assert np.allclose(agent.Q, [-0.9,  1. , -1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.allclose(agent.V, [-0.9,  1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.array_equal(agent.pi, np.arange(3))

    ## Test betamax learning.
    agent = ModelFree(policy='pessimism', eta=0.2, gamma=0.9, w=0.5)
    agent = agent.fit(gym, choice='softmax', schedule=schedule)
    assert np.allclose(agent.Q, [ 0.0,  1. , -1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.allclose(agent.V, [ 0.0,  1. ,  0. ,  0. ], atol=1e-3, rtol=0)
    assert np.array_equal(agent.pi, np.arange(3))
