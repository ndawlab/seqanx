import numpy as np
from sisyphus.envs.base import GraphWorld
from sisyphus.tests.common import test_world

def test_graph_world():
    """Test GraphWorld environment initialization."""
    
    ## Generate test gym.
    gym = GraphWorld(*test_world())
    
    ## Tests of states.
    assert np.array_equal(gym.states, np.arange(4))
    assert np.equal(gym.n_states, 4)
    assert np.array_equal(gym.viable_states, np.arange(2))
    assert np.equal(gym.n_viable_states, 2)
    
    ## Tests of info.
    assert np.array_equal(gym.info.shape, [5,4])
    assert np.array_equal(gym.info["S"].values,           [0, 1, 1, 2, 3])
    assert np.array_equal(np.concatenate(gym.info["S'"]), [1, 2, 3, 3, 2, 2, 3])
    assert np.array_equal(np.concatenate(gym.info["R"]),  [0, 1,-1,-1, 1, 0, 0])
    assert np.array_equal(np.concatenate(gym.info["T"]),  [1, 1, 0, 1, 0, 1, 1])