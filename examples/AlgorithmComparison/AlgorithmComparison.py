#python packages
#import prettyplotlib as ppl
#from prettyplotlib import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
from tqdm import tqdm, trange
from time import gmtime, strftime
import random
import gym
import random
from IPython.display import clear_output
import time
import sys

# add algorithm files to the sys path
sys.path.insert(0, '../../algorithm')
sys.path.insert(0, '../')

#user packages
import SA_early_stopping as sae
import MDP as mdp 
import MRP as mrp
import myUtils as util 



# Running three different minimax algorithms 

def algo_Compare_Exp(numExp, numIter, gamma, lambdaPar, numRestarts = 0, tqdmDisable = 1>0):
    
    # creates a two-dims toy mrp with parameter gamma and lambda
    # runs SA, SA + PR and Root numExp times. Each time uses numIter samples
    # numRestarts is the number of restarts.
    # returns 3 two-dimensinal array. Each array is of size numIter * numExp
    #
    
    # creating a two dimensional toy MRP
    
    toy_problem = mrp.two_State_MRP(lambdaPar, gamma)
    toy_Mrp, toy_MrpSampler = toy_problem.create_MRP_and_Sampler()
        
    
    # Tuning parameters for the algorithms 
    dims = (2,)
    stepOmega, stepCon = 1, 1.0 / (1 - gamma)
    optValIter = 10 * int(round(1.0/(1 - gamma)))
    Optval = toy_Mrp.get_Optval(optValIter)
    initTheta = np.zeros([2])
    burnin = int(round(1.0/(1 - gamma)**2))
    

    # Calling the algorithms
    mymethods = sae.SAmethods(dims, toy_MrpSampler, tqdmDisable = tqdmDisable)
    mymethods.set_Optval(Optval)
    
    # output will be stored here.
    errPRVals = np.zeros([numIter, numExp])
    errRootVals = np.zeros([numIter, numExp])
    errSAVals = np.zeros([numIter, numExp])
    
    #running the exps
    for i in range(numExp):
        _, errSAVals[:, i] = mymethods.SA_minimax(initTheta, numIter, gamma, stepCon = 3)
        _, errPRVals[:, i] = mymethods.SA_PR(initTheta, numIter, PROmega = 0.5, stepCon = 1, epsilon = None)
        _, errRootVals[:, i] = mymethods.ROOT(initTheta, numIter, burnin, numRestarts, epsilon = None)
    
    return errSAVals, errPRVals, errRootVals

def get_summary(data, summaryAxis):
    # data is a two dimensional array
    # output is a mean and std along summaryAxis
    
    dataMean = np.mean(data, axis = summaryAxis)
    dataStdiv = np.std(data, axis = summaryAxis) 
    
    summary = {'mean' : dataMean, 'std' : dataStdiv}
    return summary



# setting up the problem parameters
numExp, gamma, lambdaPar, numIter = 50, 0.9, 0.5, 5 * np.power(10, 5)

# setting seed
expSeed = 1234
util.set_seed(expSeed)


SAruns, PRruns, Rootruns = algo_Compare_Exp(numExp = numExp, numIter = numIter, gamma = gamma,
                                            lambdaPar = lambdaPar, tqdmDisable = 1<0)

SAsummary = get_summary(data = SAruns, summaryAxis = 1) 
PRsummary = get_summary(data = PRruns, summaryAxis = 1) 
Rootsummary = get_summary(data = Rootruns, summaryAxis = 1) 


# Saving the data


filename = 'AlgoCompare' + 'gamma_' +  str(gamma) + 'lambda_' + str(lambdaPar) + 'numExp_' + str(numExp) + '.pickle'
data = {'SA' : SAruns, 'PR': PRruns, 'Root' : Rootruns}

util.saveData(data, filename)

# logfigAlgo,_= plt.subplots(1)


# plt.loglog(SAsummary['mean'], 'g', label = 'Minimax Opt algo A', linewidth=4)
# plt.loglog(PRsummary['mean'], 'r', label = 'Minimax Opt algo B', linewidth=4)
# plt.loglog(Rootsummary['mean'], 'b', label = 'Minimax Opt algo C', linewidth=4)

# plt.legend(fontsize=15)
# plt.xlabel("Number of samples", fontsize = 18)  
# plt.ylabel("Estimation error", fontsize = 18) 
# plt.legend()
# logfigAlgo.savefig('minimax_opt_algolog.pdf' + 'gamma_' +  str(gamma) + 'lambda_'
#                    + str(lambdaPar) + 'numExp_' + str(numExp) + '.pdf', bbox_inches='tight')

