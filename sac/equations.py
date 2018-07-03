from sac.utils import profile, do_profile, transpose_csr_to_csc
import numpy as np
from pdb import set_trace as bp
import itertools

# %% LEARNING FUNCTIONS
equations = {}
equation = lambda f: equations.setdefault(f.__name__, f)


@equation
def learning_equation1(learningRate, baseLevel, activation, **kwargs):
    """ calculates the size of the increment """
    deltaB = np.multiply(activation - baseLevel, learningRate)
    return(deltaB)

@equation
def learning_equation2(learningRate, baseLevel, **kwargs):
    """ calculates the size of the increment """
    deltaB = learningRate * (1 - baseLevel)
    return(deltaB)


@equation
def learning_equation3(learningRate, baseLevel, sourceNodeAct, maxActivation, **kwargs):
    """ calculates the size of the increment """
    deltaB = learningRate * (1 - baseLevel) * sourceNodeAct / maxActivation
    return(deltaB)


@equation
def learning_equation4(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # if there are no sufficient resources for the reqired increment, reduce it
    # proportionally    
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    wmRemaining = wmRemaining + 0.000001
    deltaB = np.multiply(1 - baseLevel, learningRate)
    deltaBsum = np.sum(deltaB)
    deltaB = deltaB * min(deltaBsum, wmRemaining) / deltaBsum
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + np.sum(deltaB)
    return(deltaB)

@equation
def learning_equation5(learningRate, baseLevel, activation, maxActivation, **kwargs):
    """ calculates the size of the increment """
    deltaB = learningRate * (1 - baseLevel) * activation / maxActivation
    return(deltaB)

@equation
def learning_equation6(learningRate, baseLevel, initActivation, event, net, **kwargs):
    """ calculates the size of the increment """
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    wmRemaining = wmRemaining + 0.000001
    deltaB = np.multiply(1 - initActivation, learningRate)
    deltaB = event * deltaB
    deltaB = deltaB * min(sum(deltaB), wmRemaining) / sum(deltaB)
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + deltaB.sum()
    return(deltaB)

@equation
def learning_equation7(learningRate, baseLevel, activation, event, net, **kwargs):
    """ calculates the size of the increment """
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    wmRemaining = wmRemaining + 0.000001
    deltaB = np.multiply(1 - np.tanh(activation), learningRate)
    deltaB = event * deltaB
    deltaB = deltaB * min(sum(deltaB), wmRemaining) / sum(deltaB)
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + deltaB.sum()
    return(deltaB)

@equation
def learning_equation8(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # if there are no sufficient resources for the reqired increment, don't 
    # strenghten or create the node
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    deltaB = np.multiply(1 - baseLevel, learningRate)
    if 'CueType' in net.trials.columns:
        cueType = net.trials['CueType'].values[net.cTrial]
        if cueType == 'tbf':
            deltaB = deltaB * net.par['p_forget_prop']  
    deltaBsum = np.sum(deltaB)
    deltaB = deltaB * (deltaBsum < wmRemaining) + 0.000001 * (deltaBsum > wmRemaining)
    if deltaBsum > wmRemaining:
        spent = wmRemaining
    else:
        spent = deltaBsum
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + spent
    return(deltaB)

@equation
def learning_equation9(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # if there are no sufficient resources for the reqired increment, reduce it
    # proportionally    
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    wmRemaining = wmRemaining + 0.000001
    deltaB = np.multiply(1 - baseLevel, learningRate)
    if 'CueType' in net.trials.columns:
        cueType = net.trials['CueType'].values[net.cTrial]
        if cueType == 'tbf':
            deltaB = deltaB * net.par['p_forget_prop']  
    deltaBsum = np.sum(deltaB)
    deltaB = deltaB * min(deltaBsum, wmRemaining) / deltaBsum
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + np.sum(deltaB)
    return(deltaB)
    
@equation
def learning_equation10(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # if there are no sufficient resources for the reqired increment, reduce it
    # proportionally    
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    if wmRemaining < 0: 
        wmRemaining = 0
    deltaB = np.multiply(1 - baseLevel, learningRate)     
    wmRequested = deltaB ** net.par['w_act']
    wmReceived = wmRequested * min(np.sum(wmRequested), wmRemaining) / np.sum(wmRequested)
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + np.sum(wmReceived)
    deltaB = wmReceived ** (1/net.par['w_act'])
    return(deltaB)
    
@equation
def learning_equation11(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # if there are no sufficient resources for the reqired increment, reduce it
    # proportionally    
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    if wmRemaining < 0: 
        wmRemaining = 0
    if 'cue_type' in net.trials.columns:
        cue_type = net.trials['cue_type'].values[net.cTrial]
        if cue_type == 'tbr':
            learningRate = net.par['p_additional']
        if cue_type == 'tbf':
            learningRate = 0.01
    deltaB = np.multiply(1 - baseLevel, learningRate)  
    wmRequested = deltaB ** net.par['w_act']
    wmReceived = wmRequested * min(np.sum(wmRequested), wmRemaining) / np.sum(wmRequested)
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + np.sum(wmReceived)
    deltaB = wmReceived ** (1/net.par['w_act'])
    return(deltaB)
    
    
@equation
def learning_equation12(learningRate, baseLevel, net, **kwargs):
    """ calculates the size of the increment """
    # wm_spent is proportional to wmAvailable
    wmRemaining = net.wmAvailable[net.cTrial] - net.wmSpent[net.cTrial]
    if wmRemaining < 0: 
        wmRemaining = 0
    if 'cue_type' in net.trials.columns:
        cue_type = net.trials['cue_type'].values[net.cTrial]
        if cue_type == 'tbr':
            learningRate = net.par['p_additional']
        if cue_type == 'tbf':
            learningRate = 0.01
    deltaB = np.multiply(1 - baseLevel, learningRate)  
    wmRequested = deltaB ** net.par['w_act']
    wmReceived = wmRequested * wmRemaining / net.par['W']
    net.wmSpent[net.cTrial] = net.wmSpent[net.cTrial] + np.sum(wmReceived)
    deltaB = wmReceived ** (1/net.par['w_act'])
    return(deltaB)

@equation
def decay_power(S, T, t, d, priorB, ones):
#    bp()
    T1 = (1+t+T.data) ** d
    S1 = S.data
    if S.format == 'lil':
        S1 = list(itertools.chain.from_iterable(S1))
    try:
        T.data = S1 * T1
    except:
        bp()
    newB = transpose_csr_to_csc(T) * ones
    if type(priorB) == int:
        B = newB
    else:
        B = priorB + newB
    return(B)

@equation
def decay_exp(S, T, t, d, priorB):
    T.data = np.exp(d * (t+T.data))
    B = S.multiply(T)
    B = priorB + B * np.ones(T.shape[1])
    return(B)


def calc_preexisting_base(freqPerMillion, par):
    """ Calculates the prior base for concepts based on frequency """
    # FIXME: restimate preexisting base function
#    B = 0.8 - 0.7 * np.exp(-0.5 * freqPerMillion)
    B = par['prior0'] - par['prior1'] * np.exp(par['prior2'] * freqPerMillion)
    return(B)
    
def calc_preexisting_fan(context_diversity, par):
    """ Calculates the prior base for concepts based on frequency """
    # FIXME: restimate preexisting base function
#    B = 0.8 - 0.7 * np.exp(-0.5 * freqPerMillion)
    B = par['fan'] - par['fan'] * np.exp(1 * par['prior2'] * context_diversity)
    return(B)    


def get_equations():
    return(equations)
    
    
if __name__ == "__main__":
    import os
    os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')
    from models.inprogress.speed_testing_reder import *
