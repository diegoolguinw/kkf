import numpy as np
from numpy.linalg import pinv, cholesky
from scipy.linalg import block_diag
from covariances import compute_initial_covariance, compute_dynamics_covariance, compute_observation_covariance

from typing import Callable, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_continuous