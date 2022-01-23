import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
import random
import gym
import random




class SAmethods(object):
    
    '''Stochastic approximation methods class which contains three algorithms. Stochastic approximation,
       The stochastic approximation with Polyak Ruppert averaging, and the Recursive-one-over-t (ROOT)  
       algorithm. 
       
       Inputs: 
       
           dims: a tuple containing the dimensions. e.g. dims = (64, 4) means the algorithms will have 
                 a 64 * 4 dimensional array as inputs of the SA algorithms. 
           
           noisyOp : This a class which have two methods get_eval(theta) and get_eval_twoPoint(theta) which provides
                     one point and two point evaluation at some input point theta. The output of noisyOp.get_eval()
                     and noisyOp.get_eval_twoPoint(theta) MUST have dimension dims.
                     
                     The noisyOp.get_eval() method is always necessary for using any algorithm in this class.
                     The noisyOp.get_eval_twoPoint() is only necessary for using ROOT algorithm. 
    '''
    
    
    def __init__(self, dims, noisyOp, tqdmDisable = 1>0, addHigherOrder = 1>0):
        self.dims = dims
        self.noisyOp = noisyOp
        self.Optval = np.zeros(self.dims)
        self.tqdmDisable = tqdmDisable
        self.addHigherOrder = addHigherOrder
        
        
        
        '''getters and setters'''
    def set_dims(self, dimsNew):
        self.dims = dimsNew
            
    def set_Optval(self, OptvalNew):
        self.Optval = OptvalNew
            
            
    def get_dims(self):
        return self.dims
        
    def get_Optval(self):
        return self.Optval
    
    def set_tqdm(self, newVal):
        self.tqdm_disable = newVal
        
    def set_addHigherOrder(self, newVal):
        self.addHigherOrder = newVal
        
        
        
    def SA(self,  initTheta, numIter, stepOmega, stepCon = 1, epsilon = None):
        
        '''Stochastic approximation algorithm
        
        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)
                        
                        
            numIter = number of iterations >= 1
            
            stepsize = stepCon / (t + 1)**stepOmega, stepCon = 1 by default.
            
        
        
        Outputs:
            
            theta: the putput of the algorithm which is an array with dimension 
                   SAmethods.dims
                   
            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval
        
        '''
        
        
        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in SA algorithm')
            
        
        theta = initTheta
        thetaNew = initTheta
        estimationError = np.zeros([numIter])
        
        for i in tqdm(range(numIter), disable = self.tqdmDisable):
            noisyOpEval = self.noisyOp.get_eval(theta)
            stepsize = 1 / (1 + (i + 1) / stepCon)
            thetaNew = (1 - stepsize) * theta + stepsize * noisyOpEval
            estimationError[i] = npl.norm(theta.flatten() - self.Optval.flatten(), np.inf)
            
            theta = thetaNew
            
            if epsilon is not None:
                if estimationError[i] < epsilon:
                    return theta, estimationError[:i]            
            
        
        return theta, estimationError
    
    
    def SA_PR(self,  initTheta, numIter, PROmega, stepCon = 1, epsilon = None):
        
        '''Stochastic approximation algorithm with Polyak Ruppert averaging
        
        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)
                        
                        
            numIter = number of iterations >= 1
            
            stepsize = stepCon / (t + 1)**PROmega, stepCon = 1 by default.
            
        
        
        Outputs:
            
            thetaAvg: the putput of the algorithm which is an array with dimension 
                   SAmethods.dims
                   
            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval
        
        '''
        
        
        
        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in SA + PR algorithm')
        
        theta = initTheta
        thetaNew = initTheta
        thetaAvg = initTheta
        
        estimationError = np.zeros([numIter])
        
        for i in tqdm(range(numIter), disable = self.tqdmDisable):
            noisyOpEval = self.noisyOp.get_eval(theta)
            stepsize = stepCon / ((i + 1)**PROmega)
            thetaNew = (1 - stepsize) * theta + stepsize * noisyOpEval
            thetaAvg = (i/(i + 1)) * thetaAvg + (1/(i + 1)) * thetaNew
            theta = thetaNew
            
            estimationError[i] = npl.norm(thetaAvg.flatten() - self.Optval.flatten(), np.inf)
            
            if epsilon is not None:
                if estimationError[i] < epsilon:
                    return thetaAvg, estimationError[:i]            
            
        
        return thetaAvg, estimationError     
    
    
    def ROOT(self, initTheta, numIter, burnin, numRestarts, epsilon = None):
        
        '''ROOT-OP algorithm

        
        
        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)
                        
                        
            numIter = number of iterations >= 1
            
            Here, we use stepsize is 1/\sqrt{t}
            
            burnin = the burnin samples >= 1
            
            numRestarts = number of restarts in the ROOT-OP algorithm. 
            
            
        
        
        Outputs:
            
            thetaT: the putput of the algorithm which is an array with dimension 
                    SAmethods.dims
                   
            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval.
        
        '''
        
        
        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in ROOT algorithm')

        
        '''Doing restarts'''
        
        initTheta = self.ROOT_restarts(initTheta, burnin, numRestarts)
        
        '''The main ROOT-OP algorithm'''
        
        thetaTminusTwo = initTheta
        thetaTminusOne = initTheta
        thetaT = initTheta        
        estimationError = np.zeros([numIter])

        
        '''Initial burnin'''
        if burnin > 0:
            vtOld = np.zeros_like(initTheta)
            for i in range(burnin):
                noisyOpEval = self.noisyOp.get_eval(initTheta)
                vtOld += (noisyOpEval - initTheta)
                
            vtOld = vtOld / burnin

        else:
            print("burnin must be positive")
            return 0
        
        # Root starts from t = 1, so assigning estimatio error at t =0 
        estimationError[: burnin] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
        '''Loop after burnin'''
        
        for t in tqdm(range(burnin, numIter), disable = self.tqdmDisable):
            
            evalTminusOne, evalTminusTwo = self.noisyOp.get_twoPoint_eval(thetaTminusOne, thetaTminusTwo)
            vtNew = evalTminusOne - thetaTminusOne + ((t - 1)/t) * (vtOld - evalTminusTwo + thetaTminusTwo)    
            stepsize = 1/((t + 1)**0.5)
            thetaT = thetaTminusOne + stepsize * vtNew
            thetaTminusTwo = thetaTminusOne
            thetaTminusOne = thetaT
            vtOld = vtNew 
            
            estimationError[t] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
            
            if epsilon is not None:
                if estimationError[t] < epsilon:
                    return thetaT, estimationError[:t]
            
        
        return thetaT, estimationError
    
    def ROOT_restarts(self, initTheta, burnin, numRestarts):
        
        '''Method for doing restarts for ROOT-OP'''
        theta = initTheta
        thetaNew = initTheta
        for i in range(numRestarts):
            thetaNew, _ = self.ROOT(theta, 2* burnin, burnin, numRestarts=0)
            theta = thetaNew
            
        return theta










class MRP_SAmethods(SAmethods):
    """ SA methods subclass specifically for performing confidence interval ROOT SGD
    for MRPs """

    def __init__(self, dims, noisyOp, tqdmDisable = 1>0, addHigherOrder = 1>0):
        # calls the SAmethods initialization
        super().__init__(dims, noisyOp, tqdmDisable, addHigherOrder)


    def ROOT_MRP_with_CI(self, initTheta, numIter, burnin, gamma, numRestarts, CIPrefactor = 1,
                    pardelta = 0.1):

        '''ROOT-OP algorithm with CI for MRP.
           We also assume that states are one dimensional and are indexed

           states = {0, 1, \ldots, numStates}

            initTheta is a value function here which is a one dimensional array of length numStates
            self.dims = numStates


        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)


            numIter = number of iterations >= 1

            Here, we use stepsize is 1/\sqrt{t}

            burnin = the burnin samples >= 1

            numRestarts = number of restarts in the ROOT-OP algorithm. 




        Outputs:

            thetaT: the putput of the algorithm which is an array with dimension 
                    SAmethods.dims

            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval

        '''


        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in ROOT algorithm')
        numStates = len(initTheta)



        '''Doing restarts'''
        if numRestarts > 0:
            initTheta = self.ROOT_restarts(initTheta, burnin, numRestarts)

        '''The main ROOT-OP algorithm'''

        thetaTminusTwo = initTheta
        thetaTminusOne = initTheta
        thetaT = initTheta        
        estimationError = np.zeros([numIter])


        '''We need to store the data as we need to do confidence Intervals''' 
        getFullData = 1>0
        rewardData = np.zeros([numStates, numIter])
        transitionData = np.zeros([numStates, numStates, numIter])



        '''Initial burnin'''
        if burnin > 0:
            vtOld = np.zeros_like(initTheta)
            for i in range(burnin):
                noisyOpEval, rSample, Psample = self.noisyOp.get_eval(initTheta, getFullData)
                vtOld += (noisyOpEval - initTheta)

                rewardData[:, i] = rSample
                transitionData[:, :, i] = Psample
            vtOld = vtOld / burnin

        else:
            print("burnin must be positive")
            return 0

        # Root starts from t = 1, so assigning estimatio error at t =0 
        estimationError[: burnin] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
        '''Loop after burnin'''


        for t in tqdm(range(burnin, numIter), disable = self.tqdmDisable):

            # The self.noisyOp is returning the full matrix here as getFullData = 1>0
            evalTminusOne, evalTminusTwo, rSample, Psample = self.noisyOp.get_twoPoint_eval(thetaTminusOne, thetaTminusTwo, getFullData)
            vtNew = evalTminusOne - thetaTminusOne + ((t - 1)/t) * (vtOld - evalTminusTwo + thetaTminusTwo)    
            stepsize = 1/((t + 1)**0.5)
            thetaT = thetaTminusOne + stepsize * vtNew
            thetaTminusTwo = thetaTminusOne
            thetaTminusOne = thetaT
            vtOld = vtNew 

            #Storing the ell_infty error and the reward and transition samples. 
            estimationError[t] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
            rewardData[:, t] = rSample
            transitionData[:, :, t] = Psample



        holdoutSize =  int(round(1/(1 - gamma)**2))
        CovhatLeCam = self.get_covhat_LeCam(rewardData, transitionData, gamma, thetaT, holdoutSize)
        
        if self.addHigherOrder:
            CI_lower = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) * np.log(1. / pardelta)) / np.sqrt(max(numIter - 2*holdoutSize, 1)) 
                        
            CI_higher = (6 * np.log((8*numStates)/pardelta) * (self.span(thetaT)) / ((1 - gamma)*(numIter - 2*holdoutSize)))
            CI = CI_lower + CI_higher

        else:
            CI = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) / np.sqrt(numIter - 2*holdoutSize))


        return thetaT, estimationError, CI
    
    
    
    def get_covhat_LeCam(self, rewardData, transitionData, gamma, valueHat, holdoutSize):
        numStates = len(valueHat)

        # Estimating two independent holdout estimates of the transition matrix
        PbarOne = np.mean(transitionData[:, :, :holdoutSize])
        PbarTwo = np.mean(transitionData[:, :, holdoutSize : 2 *holdoutSize])

        #Estimating the Cov(Bellman_Op)
        _, _, numToltalData = transitionData.shape
        numCovData = numToltalData - 2 * holdoutSize
        rewardCovData, transitionCovData = rewardData[:, 2*holdoutSize :], transitionData[:, :, 2*holdoutSize :] 

        covhatBellman = np.zeros([numStates, numStates])
        avgVec = np.zeros([numStates])

        for j in tqdm(range(numCovData), disable = self.tqdmDisable):
            vec = rewardCovData[:, j]  + gamma * np.dot(transitionData[:, :, j], valueHat) 
            covhatBellman +=  np.outer(vec, vec)
            avgVec += vec 
        
        avgVec = avgVec / numCovData
        covhatBellman = (covhatBellman  - numCovData * np.outer(avgVec, avgVec)) / (numCovData - 1)
        
        # This is the leCam Cov matrix
        CovhatLeCam = (npl.inv((np.eye(numStates) - gamma * PbarOne)) @ covhatBellman) @ npl.inv((np.eye(numStates) - gamma * PbarTwo)).T

        return CovhatLeCam
    
    def span(self,vec):
        return np.max(vec) - np.min(vec)

    def get_accuracy(self, targetEpsilon, initTheta, initIter, burnin, gamma,
                     numRestarts, pardelta = 0.1, maxIter = 10000000, mult_factor = 2.0, CIPrefactor = 1.0):
    
        totalIter = initIter
        
        Val, trueErrors, CIepsilon =  self.ROOT_MRP_with_CI(initTheta = initTheta, numIter = totalIter, burnin = burnin, gamma = gamma,
                                                                  numRestarts = 0, CIPrefactor = CIPrefactor,
                                                                  pardelta=pardelta)
        
        currEpoch = 0
        horizon = 1. / (1 - gamma)

        while CIepsilon > targetEpsilon:
            totalIter = int(mult_factor * totalIter)
            pardelta = pardelta / 2.0
            
            #numRestarts = int(np.log2(totalIter * horizon))

            Val, trueErrors, newCIepsilon = self.ROOT_MRP_with_CI(initTheta = initTheta, numIter = totalIter, burnin = burnin, gamma = gamma,
                                                                  numRestarts = 0, CIPrefactor = CIPrefactor,
                                                                  pardelta=pardelta)
                  
            CIepsilon = newCIepsilon

            currEpoch = currEpoch + 1

            if totalIter > maxIter:
                print("Max iter reached")
                return Val,  trueErrors, CIepsilon, totalIter 

        return Val, trueErrors, CIepsilon, totalIter + np.power(horizon, 2) * 2



    def ROOT_MRP_with_CI_proper(self, targetEpsilon, initTheta, numIter, burnin, gamma, numRestarts, CIPrefactor = 1,
                    pardelta = 0.1, maxIter = 1000000, mult_factor = 2.0):

        '''ROOT-OP algorithm with CI for MRP.
           We also assume that states are one dimensional and are indexed

           states = {0, 1, \ldots, numStates}

            initTheta is a value function here which is a one dimensional array of length numStates
            self.dims = numStates


        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)

            targetEpsilon = desired error in infinity norm of value function

            initTheta = initialization point of the algorithm

            numIter = number of iterations >= 1

            Here, we use stepsize is 1/\sqrt{t}

            burnin = the burnin samples >= 1

            gamma = discount factor

            numRestarts = number of restarts in the ROOT-OP algorithm. 




        Outputs:

            thetaT: the output of the algorithm which is an array with dimension 
                    SAmethods.dims

            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval. This only has significance if we 
                              set the SAmethods.Optval to something.

            CI : the value of the confidence interval at the end of the algorithm

            epoch_points: the list of iterates that we compute the confidence interval for

            CI_list : a list of the confidence interval values computed at the end of each epoch

        '''


        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in ROOT algorithm')
        numStates = len(initTheta)



        '''Doing restarts'''
        if numRestarts > 0:
            initTheta = self.ROOT_restarts(initTheta, burnin, numRestarts)

        '''The main ROOT-OP algorithm'''

        thetaTminusTwo = initTheta
        thetaTminusOne = initTheta
        thetaT = initTheta        
        estimationError = np.zeros([numIter])


        '''We need to store the data as we need to do confidence Intervals''' 
        getFullData = 1>0
        rewardData = np.zeros([numStates, numIter])
        transitionData = np.zeros([numStates, numStates, numIter])



        '''Initial burnin'''
        if burnin > 0:
            vtOld = np.zeros_like(initTheta)
            for i in range(burnin):
                noisyOpEval, rSample, Psample = self.noisyOp.get_eval(initTheta, getFullData)
                vtOld += (noisyOpEval - initTheta)

                rewardData[:, i] = rSample
                transitionData[:, :, i] = Psample
            vtOld = vtOld / burnin

        else:
            print("burnin must be positive")
            return 0

        # Root starts from t = 1, so assigning estimatio error at t =0 
        estimationError[: burnin] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
        '''Loop after burnin'''


        t = burnin
        epoch_check = numIter
        holdoutSize =  int(round(1/(1 - gamma)**2))
        curr_epoch = 0
        epoch_points = [epoch_check]
        CI_list = []
        


        while t < maxIter:
            # iterate updates
            evalTminusOne, evalTminusTwo, rSample, Psample = self.noisyOp.get_twoPoint_eval(thetaTminusOne, thetaTminusTwo, getFullData)
            vtNew = evalTminusOne - thetaTminusOne + ((t - 1)/t) * (vtOld - evalTminusTwo + thetaTminusTwo)    
            stepsize = 1/((t + 1)**0.5)
            thetaT = thetaTminusOne + stepsize * vtNew
            thetaTminusTwo = thetaTminusOne
            thetaTminusOne = thetaT
            vtOld = vtNew 

            #Storing the ell_infty error and the reward and transition samples. 
            estimationError[t] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
            rewardData[:, t] = rSample
            transitionData[:, :, t] = Psample

            if t == epoch_check - 1:

                CovhatLeCam = self.get_covhat_LeCam(rewardData, transitionData, gamma, thetaT, holdoutSize)

                if self.addHigherOrder:
                    CI_lower = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) * np.log(1. / pardelta)) / np.sqrt(max(epoch_check - 2*holdoutSize, 1)) 
                    CI_higher =  (np.log((8*numStates)/pardelta) * (self.span(thetaT)) / ((1 - gamma)*(epoch_check - 2*holdoutSize)))
                    CI = CI_lower + CI_higher

                else:
                    CI = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) / np.sqrt(epoch_check - 2*holdoutSize))

                CI_list.append(CI)

                if CI <= targetEpsilon:
                    break
                else:
                    # pads all the existing arrays so the algorithm can continue

                    pardelta = pardelta / 2.0
                    new_epoch_check = int(mult_factor * epoch_check) + 1
                    curr_epoch += 1

                    new_estimation = np.zeros([new_epoch_check])
                    new_estimation[:epoch_check] = estimationError
                    estimationError = new_estimation

                    new_reward_data = np.zeros([numStates, new_epoch_check])
                    new_reward_data[:, :epoch_check] = rewardData
                    rewardData = new_reward_data

                    new_transition_data = np.zeros([numStates, numStates, new_epoch_check])
                    new_transition_data[:, :, :epoch_check] = transitionData
                    transitionData = new_transition_data

                    epoch_points.append(new_epoch_check)

                    epoch_check = new_epoch_check

            t += 1


            
        return thetaT, estimationError, CI, epoch_points, CI_list
    




class MDP_SAmethods(SAmethods):
    """ SA methods subclass specifically for performing confidence interval ROOT SGD
    for MRPs """

    def __init__(self, dims, noisyOp, tqdmDisable = 1>0, addHigherOrder = 1>0):
        # calls the SAmethods initialization
        super().__init__(dims, noisyOp, tqdmDisable, addHigherOrder)


    def ROOT_MRP_with_CI(self, initTheta, numIter, burnin, gamma, numRestarts, CIPrefactor = 3,
                    pardelta = 0.1):

        """ROOT-OP algorithm with CI for MRP.
       We also assume that states are one dimensional and are indexed

       states = {0, 1, \ldots, numStates}

        initTheta is a value function here which is a one dimensional array of length numStates
        self.dims = numStates


    Inputs: 
        initTheta = This is the initializer of the algorithm,
                    an array (single / multidimensional) with dimension
                    SAmethods.dims (the dimesion of the class)


        numIter = number of iterations >= 1

        Here, we use stepsize is 1/\sqrt{t}

        burnin = the burnin samples >= 1

        numRestarts = number of restarts in the ROOT-OP algorithm. 




    Outputs:

        thetaT: the putput of the algorithm which is an array with dimension 
                SAmethods.dims

        estimationError : This is a one dimensional array of length numIter 
                          containing infty-norm distances of the iterates of the algorithm
                          from the SAmethods.Optval

    """


        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in ROOT algorithm')
        numStates, numActions = np.shape(initTheta)



        '''Doing restarts'''
        if numRestarts > 0:
            initTheta = self.ROOT_restarts(initTheta, burnin, numRestarts)

        '''The main ROOT-OP algorithm'''

        thetaTminusTwo = initTheta
        thetaTminusOne = initTheta
        thetaT = initTheta        
        estimationError = np.zeros([numIter])


        '''We need to store the data as we need to do confidence Intervals''' 
        getFullData = 1>0
        rewardData = np.zeros([numStates, numActions, numIter])
        transitionData = np.zeros([numStates, numActions, numStates, numIter])



        '''Initial burnin'''
        if burnin > 0:
            vtOld = np.zeros_like(initTheta)
            for i in range(burnin):
                noisyOpEval, rSample, Psample = self.noisyOp.get_eval(initTheta, getFullData)
                vtOld += (noisyOpEval - initTheta)

                rewardData[:, :, i] = rSample
                transitionData[:, :, :, i] = Psample
            vtOld = vtOld / burnin

        else:
            print("burnin must be positive")
            return 0

        # Root starts from t = 1, so assigning estimatio error at t =0 
        estimationError[: burnin] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
        '''Loop after burnin'''


        for t in tqdm(range(burnin, numIter), disable = self.tqdmDisable):

            # The self.noisyOp is returning the full matrix here as getFullData = 1>0
            evalTminusOne, evalTminusTwo, rSample, Psample = self.noisyOp.get_twoPoint_eval(thetaTminusOne, thetaTminusTwo, getFullData)
            vtNew = evalTminusOne - thetaTminusOne + ((t - 1)/t) * (vtOld - evalTminusTwo + thetaTminusTwo)    
            stepsize = 1/((t + 1)**0.5)
            thetaT = thetaTminusOne + stepsize * vtNew
            thetaTminusTwo = thetaTminusOne
            thetaTminusOne = thetaT
            vtOld = vtNew 

            #Storing the ell_infty error and the reward and transition samples. 
            estimationError[t] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
            rewardData[:, :, t] = rSample
            transitionData[:, :, :, t] = Psample



        holdoutSize =  int(round(1/(1 - gamma)**2))
        CovhatLeCam = self.get_covhat_LeCam(rewardData, transitionData, gamma, thetaT, holdoutSize)
        
        if self.addHigherOrder:
            CI = (CIPrefactor * np.sqrt(np.max(CovhatLeCam)) ) / (np.sqrt(numIter) * (1 - gamma)) \
                    + (3 * np.log((8*numStates)/pardelta) * (self.span(thetaT)) / ((1 - gamma)*(numIter - 2*holdoutSize)))

        else:
            CI = (CIPrefactor * np.sqrt(np.max(CovhatLeCam)) / np.sqrt(numIter - 2*holdoutSize))

            
        return thetaT, estimationError, CI



    def ROOT_MRP_with_CI_proper(self, targetEpsilon, initTheta, numIter, burnin, gamma, numRestarts, CIPrefactor = 1,
                    pardelta = 0.1, maxIter = 1000000, mult_factor = 2.0):

        '''ROOT-OP algorithm with CI for MDP.
           We also assume that states are one dimensional and are indexed

           states = {0, 1, \ldots, numStates}

            initTheta is a value function here which is a one dimensional array of length numStates
            self.dims = numStates


        Inputs: 
            initTheta = This is the initializer of the algorithm,
                        an array (single / multidimensional) with dimension
                        SAmethods.dims (the dimesion of the class)


            numIter = number of iterations >= 1

            Here, we use stepsize is 1/\sqrt{t}

            burnin = the burnin samples >= 1

            numRestarts = number of restarts in the ROOT-OP algorithm. 




        Outputs:

            thetaT: the output of the algorithm which is an array with dimension 
                    SAmethods.dims

            estimationError : This is a one dimensional array of length numIter 
                              containing infty-norm distances of the iterates of the algorithm
                              from the SAmethods.Optval. This only has significance if we 
                              set the SAmethods.Optval to something.

            CI : the value of the confidence interval at the end of the algorithm

            epoch_points: the list of iterates that we compute the confidence interval for

            CI_list : a list of the confidence interval values computed at the end of each epoch
        '''


        if initTheta.shape != self.dims:
            raise ValueError('dimension mismatch in ROOT algorithm')
        numStates, numActions = np.shape(initTheta)



        '''Doing restarts'''
        if numRestarts > 0:
            initTheta = self.ROOT_restarts(initTheta, burnin, numRestarts)

        '''The main ROOT-OP algorithm'''

        thetaTminusTwo = initTheta
        thetaTminusOne = initTheta
        thetaT = initTheta        
        estimationError = np.zeros([numIter])


        '''We need to store the data as we need to do confidence Intervals''' 
        getFullData = 1>0
        rewardData = np.zeros([numStates, numActions, numIter])
        transitionData = np.zeros([numStates, numActions, numStates, numIter])



        '''Initial burnin'''
        if burnin > 0:
            vtOld = np.zeros_like(initTheta)
            for i in range(burnin):
                noisyOpEval, rSample, Psample = self.noisyOp.get_eval(initTheta, getFullData)
                vtOld += (noisyOpEval - initTheta)

                rewardData[:, :, i] = rSample
                transitionData[:, :, :, i] = Psample
            vtOld = vtOld / burnin

        else:
            print("burnin must be positive")
            return 0

        # Root starts from t = 1, so assigning estimatio error at t =0 
        estimationError[: burnin] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
        '''Loop after burnin'''


        t = burnin
        epoch_check = numIter
        holdoutSize =  int(round(1/(1 - gamma)**2))
        curr_epoch = 0
        epoch_points = [epoch_check]
        CI_list = []
        


        while t < maxIter:
            # iterate updates
            evalTminusOne, evalTminusTwo, rSample, Psample = self.noisyOp.get_twoPoint_eval(thetaTminusOne, thetaTminusTwo, getFullData)
            vtNew = evalTminusOne - thetaTminusOne + ((t - 1)/t) * (vtOld - evalTminusTwo + thetaTminusTwo)    
            stepsize = 1/((t + 1)**0.5)
            thetaT = thetaTminusOne + stepsize * vtNew
            thetaTminusTwo = thetaTminusOne
            thetaTminusOne = thetaT
            vtOld = vtNew 

            estimationError[t] = npl.norm(thetaT.flatten() - self.Optval.flatten(), np.inf)
            rewardData[:, :, t] = rSample
            transitionData[:, :, :, t] = Psample

            if t == epoch_check - 1:

                CovhatLeCam = self.get_covhat_LeCam(rewardData, transitionData, gamma, thetaT, holdoutSize)

                if self.addHigherOrder:
                    CI_lower = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) * np.log(1. / pardelta)) / np.sqrt(max(epoch_check - 2*holdoutSize, 1)) 
                    CI_higher =  (np.log((8*numStates)/pardelta) * (self.span(thetaT)) / ((1 - gamma)*(epoch_check - 2*holdoutSize)))
                    CI = CI_lower + CI_higher

                else:
                    CI = (CIPrefactor * np.sqrt(np.max(np.diag(CovhatLeCam))) / np.sqrt(epoch_check - 2*holdoutSize))

                CI_list.append(CI)

                if CI <= targetEpsilon:
                    break
                else:
                    # pads all the existing arrays so the algorithm can continue

                    pardelta = pardelta / 2.0
                    new_epoch_check = int(mult_factor * epoch_check) + 1
                    curr_epoch += 1

                    new_estimation = np.zeros([new_epoch_check])
                    new_estimation[:epoch_check] = estimationError
                    estimationError = new_estimation

                    new_reward_data = np.zeros([numStates, numStates, new_epoch_check])
                    new_reward_data[:, :, :epoch_check] = rewardData
                    rewardData = new_reward_data

                    new_transition_data = np.zeros([numStates, numStates, numStates, new_epoch_check])
                    new_transition_data[:, :, :, :epoch_check] = transitionData
                    transitionData = new_transition_data

                    epoch_points.append(new_epoch_check)

                    epoch_check = new_epoch_check

            t += 1
            
        return thetaT, estimationError, CI, epoch_points, CI_list
    
    def get_covhat_LeCam(self, rewardData, transitionData, gamma, QvalueHat, holdoutSize):
        numStates, numActions = np.shape(QvalueHat)


        #Estimating the Cov(Bellman_Op)
        _, _, _, numToltalData = transitionData.shape
        numCovData = numToltalData
        rewardCovData, transitionCovData = rewardData, transitionData

        covhatBellman = np.zeros([numStates, numActions])
        avgVec = np.zeros([numStates, numActions])
        bellman_values = np.zeros([numStates, numActions, numCovData])

        # loop to compute average
        for j in range(numCovData):
            vec = self.noisyOp.compute_bellman(rewardCovData[:, :, j], transitionCovData[:, :, :, j], QvalueHat)
            avgVec += vec
            bellman_values[:, :, j] = vec

        avgVec /= numCovData

        for j in range(numCovData):
            covhatBellman += np.power(bellman_values[:, :, j] - avgVec, 2)

        covhatBellman /= (numCovData - 1)

        return covhatBellman
    
    def span(self,vec):
        return np.max(vec) - np.min(vec)
    
    def get_accuracy(self, targetEpsilon, initTheta, initIter, burnin, gamma,
                     numRestarts, pardelta = 0.1, maxIter = 1000000, mult_factor = 1.2):
    
        totalIter = initIter
        
        Val, trueErrors, CIepsilon =  self.ROOT_MRP_with_CI(initTheta, totalIter, burnin, gamma, numRestarts, pardelta)
        
        currEpoch = 0
        horizon = 1. / (1 - gamma)

        while CIepsilon > targetEpsilon:
            totalIter = int(mult_factor * totalIter)
            pardelta = pardelta / 2.0
            
            #numRestarts = int(np.log2(totalIter * horizon))

            Val, trueErrors, newCIepsilon = self.ROOT_MRP_with_CI(initTheta, totalIter, burnin,
                                                                  gamma, numRestarts,pardelta)

            CIepsilon = newCIepsilon

            currEpoch = currEpoch + 1

            if totalIter > maxIter:
                print("Max iter reached")
                return Val,  trueErrors, CIepsilon, totalIter 

        return Val, trueErrors, CIepsilon, totalIter
