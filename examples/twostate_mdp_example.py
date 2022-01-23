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


# adds algorithm files to system path for importing
sys.path.insert(0, '../algorithm')

# User packages 

import MDP as mdp 
import MRP as mrp
import toyMDPexp as toyMDP
import myUtils as util 
from joblib import Parallel, delayed
import itertools


#####################################################################

def run_mdp(gamma, lambdaPar, targetEpsilon):
	# helper function that runs the trial with parameters

	toyMDP.runToy(gamma, lambdaPar, targetEpsilon, num_trials, expSeed, 
						outdir, numRestarts = 0, plugin = True)

#####################################################################

# Output folder where the data is saved 

outdir = 'toyMDP/'
overwriteData = False

# checks to overwrite or not

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


# Trial inputs
num_trials = 10 # sets the number of trials
gamma_vals = 1 - np.power(10, -np.linspace(1, 2, 5)) # sets the range of discount factor
lambda_vals = [0.5, 1.0, 1.5] # controls difficulty -- 0.0 is the hardest problem. lambda large is easy problem
eps_vals = [0.05] # sets the tolerance


num_cores = 5 # number of cores to use

# parallelize running the mrp, save files to the directory specified
Parallel(n_jobs = num_cores)(delayed(run_mdp)(gamma, lamb, eps) \
                    for gamma, lamb, eps in itertools.product(gamma_vals, lambda_vals, eps_vals))
