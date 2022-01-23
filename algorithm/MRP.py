import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
import random
import gym
import random




class MRP(object):
    
    '''Creates MRP with reward, transition and gamma
    
    Inputs: 
        reward: One dimensional array. e.g. reward[state] 
        
        transition is a two dimensional array. e.g. transition[state, nextState]
        
        gamma is the discount actor \in (0, 1)
    '''
    
    def __init__(self, reward, transition, gamma):
        self.reward = reward
        self.transition = transition
        self.gamma = gamma
        self.numStates, _ = transition.shape 
        
        
    
    '''getters'''
    def get_reward(self):
        return self.reward
    
    def get_transition(self):
        return self.transition
    
    def get_gamma(self):
        return self.gamma
    
    def get_numStates(self):
        return self.numStates
        
    
    '''setters'''
    
    def set_reward(self, reward):
        self.reward = reward
        
    def set_transition(self, transition):
        self.transition = transition
    
    def set_gamma(self, gamma):
        self.gamma = gamma 
        
    '''bellman update for policy evaluaton'''
        
    def bellman_Update(self, Val):
        return self.reward + self.gamma * np.dot(self.transition, Val)
                
    '''Calculating the value-function.'''
    
    def get_Optval(self, numIter, initVal = None):
        
        ''' 
        Inputs: 
        
            numIter: number of Bellman iterations to be performned
        
            initQval: initial value of the Q-function, it is a two dimensional array.
                      initQval.shape = numStates, numActions 
        
        The output is a two dimensional array --> Qopt[state, action]
           
           
        '''
        
        if initVal is None:
            initVal = np.zeros_like(self.reward)
        
        
        Val = initVal
        for i in range(numIter):
            ValNew = self.bellman_Update(Val)
            Val = ValNew
            
        self.Optval = Val
        return self.Optval
    
    
    
    

class MRP_sampler(MRP):
    
     
    '''Creates MRP sampler with reward, reward std, transition and gamma
    
    Inputs: 
        reward: One dimensional array. e.g. reward[state] 
        rewardStd: One dimensional array with reward standard deviation. e.g. rewardStd[state]
        
        transition is a two dimensional array. e.g. transition[state, nextState]
        
        gamma is the discount actor \in (0, 1)
        
        
        Outputs:
        
        creates the class which can be used to generate generative samplers from the MRP
        
        get_eval(theta) ----> returns one point evaluation at theta
        
        get_twoPoint_eval(thetaOne, thetaTwo) ---> returns two point evaluation at thetaOne and thetaTwo 
        
    '''
    
    
    def __init__(self, reward, transition, gamma, rewardStd):
        MRP.__init__(self, reward, transition, gamma)
        self.rewardStd = rewardStd
        
        
        '''new getters and setters'''
    def get_rewardStd(self):
        return self.rewardStd
    
    def set_rewardStd(self, newRewardStd):
        self.rewardStd = newRewardStd
        
        
     
    def get_transition_and_reward_samples(self):
        '''Method for obtaining one  generative sample of reward and transition'''  
        rewardSample = np.zeros_like(self.reward)
        transitionSample = np.zeros_like(self.transition)
        
        for state in range(self.numStates):
            rewardSample[state] = np.random.normal(self.reward[state], self.rewardStd[state])
            nextState = np.random.choice(self.numStates, 1, p = self.transition[state, :])
            transitionSample[state, nextState] = 1.0
                
        return rewardSample, transitionSample
    
    
       
    def plugin(self, numSamples):

        reward_avg = np.zeros_like(self.reward)
        transition_avg = np.zeros_like(self.transition)

        dim = np.shape(reward_avg)[0]

        for i in range(numSamples):
            reward, transition = self.get_transition_and_reward_samples()
            reward_avg += reward
            transition_avg += transition

        reward_avg /= numSamples
        transition_avg /= numSamples
        
        theta_plugin = np.dot(npl.inv(np.eye(dim) - self.gamma*transition_avg), reward_avg)
        return  theta_plugin
    
    
    def get_noisy_bellman(self, val, getFullData = 1<0):
        '''The noisy bellman update evaluated at initVal'''
        
        rewardSample, transitionSample = self.get_transition_and_reward_samples()
        
        if getFullData:
            return rewardSample + self.gamma * np.dot(transitionSample, val), rewardSample, transitionSample
        else:
            return rewardSample + self.gamma * np.dot(transitionSample, val)
        
        
    
    def get_noisy_bellman_twopoint(self, valOne, valTwo, getFullData = 1<0):
        '''The noisy bellman update evaluated at QvalOne and QvalTwo with same samples. This gives two point evaluation'''
        
        rewardSample, transitionSample = self.get_transition_and_reward_samples()
        
        bellmanOne = rewardSample + self.gamma * np.dot(transitionSample, valOne)
        bellmanTwo = rewardSample + self.gamma * np.dot(transitionSample, valTwo)
        
        if getFullData:
            return bellmanOne, bellmanTwo, rewardSample, transitionSample
        else:
            return bellmanOne, bellmanTwo
        
        
    
    def get_eval(self, val, getFullData = 1<0):
        '''One point evaluation'''
        return self.get_noisy_bellman(val, getFullData)
    
    def get_twoPoint_eval(self, valOne, valTwo, getFullData = 1<0):
        '''Two point evaulation'''
        return self.get_noisy_bellman_twopoint(valOne, valTwo, getFullData)


class two_State_MRP(object):
    '''Creates two state toy MRP from Sec 3.2.1. of the paper https://arxiv.org/pdf/2003.07337.pdf 
    
    Inputs: 
        lambdaPar : nonnegative scalar controlling the difficulty of the problem.
                    lambdaPar = 0 is the worst case problem, problem difficulty decreses as lambdaPar increases.
        
    Output:
        create_MRP_and_Sampler() gives outputs the toy MRP and a sampler  
        
        MRP and MRP_sampler class 
        
    '''
    
    def __init__(self, lambdaPar, gamma):
        self.lambdaPar = lambdaPar
        self.gamma = gamma
        
    '''getters and setters'''
    
    def get_lambdaPar(self):
        return self.lambdaPar
    
    def get_gamma(self):
        return self.gamma
    
    
    def set_lambdaPar(self, lambdaNew):
        self.lambdaPar = lambdaNew
    
    def set_gamma(self, gammaNew):
        self.gamma = gammaNew
    
    
    def create_MRP_and_Sampler(self):
        
        tau = 1 - np.power( (1 - self.gamma), self.lambdaPar)
        reward = np.array([1, tau])
        rewardStd = np.zeros_like(reward)
        
        p = (4.0*self.gamma - 1) / (3.0 * self.gamma) 
        transition = np.array([[ p, 1-p ],[0, 1]])
        
        return MRP(reward, transition, self.gamma), MRP_sampler(reward, transition, self.gamma, rewardStd)


class highdim_MRP(object):

    def __init__(self, lambdaPar, gamma, dimensions):
        self.lambdaPar = lambdaPar
        self.gamma = gamma
        self.dimensions = dimensions
        
    '''getters and setters'''
    
    def get_lambdaPar(self):
        return self.lambdaPar
    
    def get_gamma(self):
        return self.gamma
    
    def get_dimensions(self):
        return self.dimensions
    
    def set_lambdaPar(self, lambdaNew):
        self.lambdaPar = lambdaNew
    
    def set_gamma(self, gammaNew):
        self.gamma = gammaNew


    def create_MRP_and_Sampler(self):
        
        tau = 1 - np.power( (1 - self.gamma), self.lambdaPar)
        reward = np.array([1, tau])
        rewardStd = np.zeros(self.dimensions)

        full_reward = np.zeros(self.dimensions)
        full_transition = np.zeros((self.dimensions, self.dimensions))
        
        p = (4.0*self.gamma - 1) / (3.0 * self.gamma) 

        transition = np.array([[ p, 1-p ],[0, 1]])


        for i in range(self.dimensions // 2):
            full_reward[2 * i] = reward[0]
            full_reward[2 * i + 1] = reward[1]

            full_transition[2 * i, 2 * i] = transition[0, 0]
            full_transition[2 * i, 2 * i + 1] = transition[0, 1]
            full_transition[2 * i + 1, 2 * i] = transition[1, 0]
            full_transition[2 * i + 1, 2 * i + 1] = transition[1, 1]
        
        return MRP(full_reward, full_transition, self.gamma), MRP_sampler(full_reward, full_transition, self.gamma, rewardStd)

