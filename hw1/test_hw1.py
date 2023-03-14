# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import sys

import warnings
from run_hw1 import my_app
import json
import hydra


# @hydra.main(config_path="hw1/conf", config_name="config")

from hydra import compose, initialize

class TestHomeWork1(object):

    # @pytest.mark.timeout(600)
    def test_reward(self):
        """
        Check that the BC learned policy can match the expert to 30% expert reward
        """
        ### Load hydra manually.
        initialize(config_path="./conf", job_name="test_app")
        cfg = compose(config_name="config", overrides=["db.user=me"])
        returns = my_app(cfg)
        # returns = [1.0, 3.0]
        print ("returns: ", returns)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(returns) > -1.5

if __name__ == '__main__':
    pytest.main([__file__])