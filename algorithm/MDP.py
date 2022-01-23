
import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
import random
import gym
import random





class MDP(object):
    
    '''Creates MDP with reward, transition and gamma
    
    Inputs: 
        reward: Two dimensional array. e.g. reward[state, action] 
        
        transition is a three dimensional array. e.g. transition[state, action, nextState]
        
        gamma is the discount actor in (0, 1)
    '''
    
    def __init__(self, reward, transition, gamma):
        self.reward = reward
        self.transition = transition
        self.gamma = gamma
        self.numStates, self.numActions = reward.shape 
        
        
    
    '''getters'''
    def get_reward(self):
        return self.reward
    
    def get_transition(self):
        return self.transition
    
    def get_gamma(self):
        return self.gamma
    
    def get_numStates(self):
        return self.numStates
        
    def get_numActions(self):
        return self.numActions
    
    '''setters'''
    
    def set_reward(self, reward):
        self.reward = reward
        
    def set_transition(self, transition):
        self.transition = transition
    
    def set_gamma(self, gamma):
        self.gamma = gamma 
        
    '''bellman update with no noise'''
        
    def bellman_Opt_Update(self, Qval):
        
        maxFutureQ = np.zeros([self.numStates])
        for state in range(self.numStates):
            maxFutureQ[state] = np.max(Qval[state, :])
        
        for state in range(self.numStates):
            for action in range(self.numActions):
                Qval[state, action] = self.reward[state, action] + self.gamma * np.dot(self.transition[state, action, :], maxFutureQ)
                
    '''Calculating the optimal Q-function.'''
    
    def get_Qopt(self, numIter, initQval = None):
        
        ''' 
        Inputs: 
        
            numIter: number of Bellman iterations to be performned
        
            initQval: initial value of the Q-function, it is a two dimensional array.
                      initQval.shape = numStates, numActions 
        
        The output is a two dimensional array --> Qopt[state, action]
           
           
        '''
        
        if not initQval:
            initQval = np.zeros([self.numStates, self.numActions])
        
        
        Qval = initQval
        for i in range(numIter):
            QvalNew = self.bellman_Opt_Update(Qval)
            QvalNew = Qval
            
        self.Qopt = Qval
        return self.Qopt

    def get_Optval(self, numIter, initQval = None):
        return self.get_Qopt(numIter, initQval)



class MDP_sampler(MDP):
    def __init__(self, reward, transition, gamma, rewardStd):
        MDP.__init__(self, reward, transition, gamma)
        self.rewardStd = rewardStd
        
        
        '''New getters and setters'''
    def get_rewardStd(self):
        return self.rewardStd
    
    def set_rewardStd(self, rewardStdNew):
        self.rewardStd = rewardStdNew
        
     
    def get_transition_and_reward_samples(self):
        '''Method for obtaining one  generative sample of reward and transition'''  
            
        rewardSample = np.zeros_like(self.reward)
        transitionSample = np.zeros_like(self.transition)
        
        for state in range(self.numStates):
            for action in range(self.numActions):
                rewardSample[state, action] = np.random.normal(self.reward[state, action], self.rewardStd[state, action])
                nextState = np.random.choice(self.numStates, 1, p = self.transition[state, action, :])
                transitionSample[state, action, nextState] = 1.0
                
        return rewardSample, transitionSample
    

    def plugin(self, numSamples, numIter = 100):
        reward_avg = np.zeros_like(self.reward)
        transition_avg = np.zeros_like(self.transition)

        Qest = np.zeros_like(self.reward)

        for i in range(numSamples):
            reward, transition = self.get_transition_and_reward_samples()
            reward_avg += reward
            transition_avg += transition

        reward_avg /= numSamples
        transition_avg /= numSamples
        
        for i in range(numIter):
            Qest = self.compute_bellman(reward_avg, transition_avg, Qest)

        return Qest
    
    
    def get_noisy_bellman(self, initQval, getFullData = 1<0):
        '''The noisy bellman update evaluated at initQval'''
        
        rewardSample, transitionSample = self.get_transition_and_reward_samples()
        bellman = np.zeros_like(self.reward)
        
        maxFutureQ = np.zeros([self.numStates])
        
        for state in range(self.numStates):
            maxFutureQ[state] = np.max(initQval[state, :])
        
        for state in range(self.numStates):
            for action in range(self.numActions):
                bellman[state, action] = rewardSample[state, action] + self.gamma * np.dot(transitionSample[state, action, :], maxFutureQ)
    
        if getFullData:
            return bellman, rewardSample, transitionSample
        else:
            return bellman 
    
    def get_noisy_bellman_twopoint(self, QvalOne, QvalTwo, getFullData = 1<0):
        '''The noisy bellman update evaluated at QvalOne and QvalTwo with same samples. This gives two point evaluation'''
        
        rewardSample, transitionSample = self.get_transition_and_reward_samples()
        bellmanOne = np.zeros_like(self.reward)
        bellmanTwo = np.zeros_like(self.reward)
        
        maxFutureQOne = np.zeros([self.numStates])
        maxFutureQTwo = np.zeros([self.numStates])
        
        
        for state in range(self.numStates):
            maxFutureQOne[state] = np.max(QvalOne[state, :])
            maxFutureQTwo[state] = np.max(QvalTwo[state, :])
            
        
        for state in range(self.numStates):
            for action in range(self.numActions):
                bellmanOne[state, action] = rewardSample[state, action] + self.gamma * np.dot(transitionSample[state, action, :], maxFutureQOne)
                bellmanTwo[state, action] = rewardSample[state, action] + self.gamma * np.dot(transitionSample[state, action, :], maxFutureQTwo)
                
        if getFullData:
            return bellmanOne, bellmanTwo, rewardSample, transitionSample
        else:
            return bellmanOne, bellmanTwo
        
    
    def get_eval(self, Qval, getFullData = 1<0):
        '''One point evaluation'''
        return self.get_noisy_bellman(Qval, getFullData)
    
    def get_twoPoint_eval(self, QvalOne, QvalTwo, getFullData = 1<0):
        '''Two point evaulation'''
        return self.get_noisy_bellman_twopoint(QvalOne, QvalTwo, getFullData)
    
    def compute_bellman(self, rewardSample, transitionSample, Qval):
        maxFutureQ = np.zeros([self.numStates])
        bellman = np.zeros_like(self.reward)

        for state in range(self.numStates):
            maxFutureQ[state] = np.max(Qval[state, :])

        for state in range(self.numStates):
            for action in range(self.numActions):
                bellman[state, action] = rewardSample[state, action] + self.gamma * np.dot(transitionSample[state, action, :], maxFutureQ)

        return bellman        
    
    
    
    
    
    


class env_to_MDP(object):
    
    '''Takes on openai environment and extracts reward, transition and reward variance.
        This is written for the 'FrozenLake' environment, check compatibility before using elsewhere. 
    '''
    
    def __init__(self, env):
        self.env = env
        self.numStates = env.env.nS
        self.numActions = env.env.nA
    
    def extract_reward_and_transition(self):
        info = self.env.env.P
        self.reward = np.zeros([self.numStates, self.numActions])
        self.rewardStd = np.zeros([self.numStates, self.numActions])
        self.transition = np.zeros([self.numStates, self.numActions, self.numStates])
        
        
        for state in range(self.numStates):
            for action in range(self.numActions):
                numNextStates = len(info[state][action])
                
                #nextStateInfo[.][0] -- Transition prob to state s'
                #nextStateInfo[.][1] -- s'
                #nextStateInfo[.][2] -- reward(state, action, state')
                #nextStateInfo[.][3] -- is state' is Goal state
                
                for j in range(numNextStates):
                    nextState = info[state][action][j][1]

                    prob = info[state][action][j][0]
                    tempReward =  info[state][action][j][2]
                    self.reward[state, action] += prob * tempReward
                    self.rewardStd[state, action] = prob * (1 - prob) * tempReward**2 
                    self.transition[state, action, nextState] += prob 
                    
    def get_transition(self):
        return self.transition
    
    def get_reward(self):
        return self.reward
    
    def get_rewardStd(self):
        return self.rewardStd
    
    def set_reward(self, reward):
        self.reward = reward
    
    def set_transition(self, transition):
        self.transition = transition





class two_State_MDP(object):
    # Creates two state toy MDP
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
    
    
    def create_MDP_and_Sampler(self):

        transition = np.zeros((2, 2, 2))
        
        tau = 1 - np.power( (1 - self.gamma), self.lambdaPar)
        reward = np.array([[1, 0], [tau, 0]])
        rewardStd = np.zeros_like(reward)
        
        p = (4.0*self.gamma - 1) / (3.0 * self.gamma) 
        
        transition[0, 0, :] = np.array([p, 1 - p])
        transition[0, 1, :] = np.array([1, 0])
        transition[1, 0, :] = np.array([0, 1])
        transition[1, 1, :] = np.array([0, 1])
        
        return MDP(reward, transition, self.gamma), MDP_sampler(reward, transition, self.gamma, rewardStd)
