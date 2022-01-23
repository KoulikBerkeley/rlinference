import MDP as mdp 
import MRP as mrp
# import MRP_SA_early_stopping as mrpSAstop
import myUtils as util
from SA_early_stopping import MDP_SAmethods as mdpSAstop


import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
import random
import random
import time
import sys


#####################################################################

def do_stopping_exp_toyMDP(lambdaPar, gamma, targetEpsilon, numRestarts = 0, 
                                    addHigherOrder = 1>0, plugin = False, pardelta = 0.1,
                                    CIPrefactor = 2.0):

    '''
    Runs the stopping algorithm that involves estimating the confidence interval
    for solving the two-state MDP with discount gamma and lambdaPar for a target
    accuracy of targetEpsilon

    Inputs:

        lambdaPar: float describing the lambda parameter which controls the
                   difficulty of the problem

        gamma: float representing the discount parameter

        targetEpsilon: float representing the targetted error bound in ell_infty norm

        numRestarts: int, parameter for ROOT-SA algorithm controlling the number 
                     of restarts for the algorithm

        addHigherOrder: bool indicating the use of higher-order (i.e. 1/N) terms in the
                        data-dependent guarantee in EmpIRE

        plugin: bool representing using an initialization thats estimated via using
                2 / (1 - gamma)^2 smaples for a plug-in estimate

        pardelta: float for the high probability guarantee 1 - pardelta

        CIPrefactor: constant factor in front of dominating term in confidence guarantee,
                     smaller CIPrefactor means faster convergence. the theory ensures
                     setting it to be 5 ensures an accurate result, but often its a conservative
                     guarantee

    Outputs:

        Val: numpy matrix of dimension (2, 2), the estimate of the value function after running EmpIRE
        
        trueErrors: numpy array containing the error in ell-infty norm after every iteration

        CIEpsilon: float denoting the estimated error at the end of the algorithm

        totalIter: int for the number of samples/iterations required in running this procedure

    '''

    toy_problem = mdp.two_State_MDP(lambdaPar, gamma)
    toy_Mdp, toy_MdpSampler = toy_problem.create_MDP_and_Sampler()

    r = toy_MdpSampler.get_reward()
    P = toy_MdpSampler.get_transition()
    gamma = toy_MdpSampler.get_gamma()
    rStd = toy_MdpSampler.get_rewardStd()
    
    dims = (2, 2)

    if plugin:
        horizon = int(1. / (1 - gamma))
        initTheta = toy_MdpSampler.plugin(2 * np.power(horizon, 2), 10 * horizon)
    else:
        initTheta = np.zeros(dims)
    
    stepOmega, stepCon = 1, 1.0 / (1 - gamma)
    optValIter = 10 * int(round(1.0/(1 - gamma)))
    Optval = toy_Mdp.get_Optval(optValIter)

    
    mymethods = mdpSAstop(dims, toy_MdpSampler, tqdmDisable = 1>0, addHigherOrder = addHigherOrder)
    mymethods.set_Optval(Optval)
    burnin = int(round(1/(1 - gamma)**2))
    
    initIter = 4* burnin
    Val, trueErrors, CIepsilon, epoch_points, _ = mymethods.ROOT_MRP_with_CI_proper(targetEpsilon, initTheta, initIter, burnin,
                                                                   gamma, numRestarts, mult_factor = 2.0, pardelta = pardelta,
                                                                   CIPrefactor = CIPrefactor)

    totalIter = epoch_points[-1]

    if plugin:
        totalIter += int(2 * np.power(horizon, 2))

    return Val, trueErrors, CIepsilon, totalIter



def runToy(gamma, lambdaPar, targetEpsilon, numExp, expSeed, outdir, numRestarts = 5, plugin = False):


    ''' 
    Sets up running EmpIRE but also saves the results into pickle file

    Inputs:

        gamma: float representing the discount parameter

        lambdaPar: float describing the lambda parameter which controls the
                   difficulty of the problem

        targetEpsilon: float representing the targetted error bound in ell_infty norm

        numExp: int denoting the number of trials to be run

        outdir: file directory indicating what folder to be saved in

        numRestarts: int, parameter for ROOT-SA algorithm controlling the number 
                     of restarts for the algorithm

        plugin: bool representing using an initialization thats estimated via using
                2 / (1 - gamma)^2 smaples for a plug-in estimate

        pardelta: float for the high probability guarantee 1 - pardelta                

    '''



    finalTrueErrors = np.zeros([numExp])
    finalPredictedErrors = np.zeros([numExp]) 
    iterUsed = np.zeros([numExp])
    factorSaving = np.zeros([numExp])
    
    fullErrors = {}
    ValEsts = {}

    
    worstCaseIter = int(round(1/(targetEpsilon**2 * (1 - gamma)**3 ))) 

    for exp in tqdm(range(numExp)):
        Val, error, CI, totalIter =  do_stopping_exp_toyMDP(lambdaPar, gamma, targetEpsilon, numRestarts, addHigherOrder=1>0, plugin = plugin)
        
        finalTrueErrors[exp] = error[-1]
        finalPredictedErrors[exp] = CI
        iterUsed[exp] = totalIter

        factorSaving[exp] = worstCaseIter / totalIter
        
        fullErrors[exp] = error
        ValEsts[exp] = Val
    
    # data to be saved from the experiment
    data = {}
    data['fullErrors'] = fullErrors
    data['ValEsts'] = ValEsts
    data['factorSaving'] = factorSaving
    data['finalPredictedErrors'] = finalPredictedErrors
    data['finalTrueErrors'] = finalTrueErrors
    data['gamma'] = gamma
    data['lambdaPar'] = lambdaPar
    data['targetEpsilon'] = targetEpsilon
    data['numExp'] = numExp

    
    #file where the data is saved
    filename = 'toyMDP_' + str(gamma) +  '_lambdaPar_'  +   str(lambdaPar) + '_numExp_' + str(numExp) + \
    '_seed_' + str(expSeed) + '.pickle'
    
    #saving data 
    util.saveData(data, outdir +  filename)
