import numpy as np
import os
import random
import pickle



def set_seed(expSeed):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(expSeed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(expSeed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(expSeed)
    
def saveData(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def openData(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data