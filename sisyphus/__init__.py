"""Code associated with paper"""

__version__ = '0.1'

from .envs import (CliffWalking, FlightInitiationDistance, FreeChoice, 
                   OpenField, DecisionTree)
from .dp import (ValueIteration)
from .td import (ModelFree)
from .misc import (softmax,betamax)