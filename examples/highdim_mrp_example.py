import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
from tqdm import tqdm, trange
from time import gmtime, strftime
import random
import gym
from IPython.display import clear_output
import time
import sys
import pickle
import os


# add algorithm files to the sys path
sys.path.insert(0, '../algorithm')

# User packages 

import MDP as mdp 
import MRP as mrp
import highdimMRPexp as highdim

import myUtils as util 
from joblib import Parallel, delayed
import itertools


#####################################################################

# parallelized version
def run_mrp(gamma, lambdaPar, targetEpsilon, dims):
    highdim.run_highdim(gamma, lambdaPar, targetEpsilon, num_trials, 
                            expSeed, outdir, dimensions = dims, numRestarts = 0, plugin = True)


#####################################################################


# Output folder where the data is saved 
'''overwrite is turned off. Set to 1>0 if you want to overwrite '''

outdir = 'highdimMRP/'
overwriteData = True

if not overwriteData:
    os.mkdir(outdir)
    
else:
    try:
        os.mkdir(outdir)
    except Exception:
        print(' ')
        print("CAUTION: Folder already exists. You decided to overwrite")
        print(' ')

#setting seed of the experiment
expSeed = 1234
util.set_seed(expSeed)



# Inputs

num_trials = 5
gamma_vals = [0.95] # discount factor 
lambda_vals = [1.5] # controls difficulty -- 0.0 is the hardest problem. lambda large is easy problem
eps_vals = [0.1]
dimensions = [10, 50, 100, 200, 300]


num_cores = 5 # number of cores to use

# parallelize running the mrp, save files to the directory specified
Parallel(n_jobs = num_cores)(delayed(run_mrp)(gamma, lamb, eps, dims) \
                    for gamma, lamb, eps, dims in itertools.product(gamma_vals, lambda_vals, eps_vals, dimensions))
