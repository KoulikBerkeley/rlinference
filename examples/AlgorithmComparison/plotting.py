#python packages
import matplotlib.pyplot as plt
import matplotlib
# import prettyplotlib as ppl
# from prettyplotlib import brewer2mpl
import numpy as np
import numpy.linalg as npl
from tqdm import tqdm, trange
from time import gmtime, strftime
import random
import gym
import random
from IPython.display import clear_output
import time
import sys
import pandas as pd
import matplotlib.ticker as mticker

# add algorithm files to the sys path
sys.path.insert(0, '../../algorithm')
sys.path.insert(0, '../')

#user packages
import SA_early_stopping as sae
import MDP as mdp 
import MRP as mrp
import myUtils as util 


#######################################################

def get_summary(data, summaryAxis):
    # data is a two dimensional array
    # output is a mean and std along summaryAxis
    
    dataMean = np.mean(data, axis = summaryAxis)
    dataStdiv = np.std(data, axis = summaryAxis) 
    
    summary = {'mean' : dataMean, 'std' : dataStdiv}
    return summary

# setting up the problem parameters
numExp, gamma, lambdaPar, numIter = 50, 0.9, 0.5, 5 * np.power(10, 5)

filename = 'AlgoCompare' + 'gamma_' +  str(gamma) + 'lambda_' + str(lambdaPar) + 'numExp_' + str(numExp) + '.pickle'

data = pd.read_pickle(filename)

SAruns = data["SA"]
PRruns = data["PR"]
Rootruns = data["Root"]


SAsummary = get_summary(data = SAruns, summaryAxis = 1) 
PRsummary = get_summary(data = PRruns, summaryAxis = 1) 
Rootsummary = get_summary(data = Rootruns, summaryAxis = 1) 

logfigAlgo, ax= plt.subplots(1)

for i in range(numIter - 1, 0, -1):
    if SAsummary['mean'][i] >= 0.1:
        print("SA", i)
        break

for i in range(numIter - 1, 0, -1):
    if PRsummary['mean'][i] >= 0.1:
        print("PR", i)
        break

for i in range(numIter - 1, 0, -1):
    if Rootsummary['mean'][i] >= 0.1:
        print("Root", i)
        break

plt.loglog(Rootsummary['mean'], 'b', label = 'TD + Variance Reduction', linewidth=4)
plt.loglog(PRsummary['mean'], 'r', label = 'TD + PR averaging', linewidth=4)
plt.loglog(SAsummary['mean'], 'g', label = 'TD', linewidth=4)

plt.hlines(y = 0.1, xmin = 0, xmax = np.power(10, 6), 
           color = 'orange', linestyle = "--", label = "Target Error")


plt.legend(fontsize=15)

ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xticks(rotation = 45)

plt.xlabel("Number of samples", fontsize = 18)  
plt.ylabel("Estimation error", fontsize = 18) 
plt.title("Behavior of different minimax algorithms", fontsize = 25)
plt.legend()
logfigAlgo.savefig('minimax_opt_algolog.pdf' + 'gamma_' +  str(gamma) + 'lambda_'
                   + str(lambdaPar) + 'numExp_' + str(numExp) + '.pdf', bbox_inches='tight')

