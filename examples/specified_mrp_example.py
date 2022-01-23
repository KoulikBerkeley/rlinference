# Importing libraries from this package
import MRP as mrp
from SA_early_stopping import MRP_SAmethods as mrpSAstop

# Importing external libraries
import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
import random
import random
import time
import sys


########################################
# Initializes the MRP
reward = np.array([0.5, 0.7, 0.1]) # reward vector
transition = np.array([[0.5, 0.3, 0.2], # transition matrix
					   [0.1, 0.7, 0.2],
					   [0.3, 0.1, 0.6]])
gamma = 0.9 # initializes discount factor
rewardStd = np.array([0.2, 0.25, 0.1]) # standard deviation of the observed reward
dims = (3,) # sets the dimension of the problem

# Initializes the MRP sampler needed for the ROOT algorithm
mrp_sampler = mrp.MRP_sampler(reward, transition, gamma, rewardStd)

# Sets parameters for the class of SA methods
method = mrpSAstop(dims, mrp_sampler, 
	tqdmDisable = True, addHigherOrder = True)

# Sets the initial set of iterations for the algorithm
burnin = int(round(1 / np.power(1 - gamma, 2)))
initIter = 4 * int(round(1 / np.power(1 - gamma, 2)))

# Sets the initial iteration via plug-in estimates
initTheta = mrp_sampler.plugin(int(2 * np.power(1 / (1 - gamma), 2)))

# sets other parameters
target_eps = 0.1 # target accuracy
num_restarts = 0 # number of restarts for ROOT, necessary for zero initialization
CIPrefactor = 1.0 # prefactor for confidence interval term 

# runs the algorithm, second argument only has meaning if know the true value function
Val, _, CIepsilon, epoch_points, _ = method.ROOT_MRP_with_CI_proper(target_eps, initTheta, initIter, burnin,
                                                               gamma, numRestarts = 5, mult_factor = 2.0, pardelta = 0.1,
                                                               CIPrefactor = CIPrefactor)

# Outputs the result of the algorithm
print(Val, CIepsilon, epoch_points)
# prints the true value function for comparison
print(mrp_sampler.get_Optval(10 * int(round(1.0 / (1 - gamma)))))