# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 21:24:41 2017

@author: venci
"""

def build_time_matrix(trials, scale):
    """
    Create a time matrix to hold when events occured
    """
    T = trials.loc[:, ('list', 'stim', 'trial', 'onset')]
    T = T.melt(id_vars=['trial', 'onset'])
    T.onset = T.onset/scale
    T = T.pivot(index='value', columns='trial', values='onset')
    return(T)


def pprint(x):
    '''
    prints a dictionary with indented levels for readability
    can be used to print the contents of the network and the nodes
    '''

    print(json.dumps(x, indent=4))


def eval_base(S, T, cTime, priorB=0, d=0.2):
    """
    Calculates the current base-level activation of all nodes, based on the
    time of occurences (T), the increments at each time (S), and the current
    time (cTime).

    Parameters
    -----------
    T : matrix
        a matrix where each row is a single sitmulus vector, and each column
        a single trial vector. Cells contain the timestamp of when the event
        occured (if only one stimulus is displayed on a trial, it is a column
        with 0s for all other stimuli, and the timing for the presented stim)
    S : similar to T in shape and size, but stores the size of the increment at
        each occurence
    cTime: a scalar for the current time
    priorB: prexisting base-level activation. Default is zero, but you can
        supply a pandas series with the calculated prior base for concept nodes
    """
    # FIXME: Put d par in globals (load before function defs)
    # Create a forgetting matrix by which to multiply the increments
    F = np.power(1+cTime-T, -d)

    # The base is the sum of all increments, where each increment is reduced by
    # the forgetting function F
    B = np.multiply(S, F)
    B = np.nansum(B, axis=1).A1
    return(priorB+B)


def eval_links(S, T, cTime, priorB=0, d=0.2):
    """
    Calculates the current strength of all links, based on the
    time of occurences (T), the increments at each time (S), and the current
    time (cTime).

    Parameters
    -----------
    T : a 3d array where each row is a single time vector, and the 2nd and 3rd
    dimensions are a square stim matrix. A positive value means that the  items
    where experienced together. Cells contain the timestamp of when the event
    S : similar to T in shape and size, but stores the size of the increment at
        each occurence
    cTime: a scalar for the current time
    priorB: prexisting strength. Default is zero, but you can supply
            a pandas series with the calculated prior bases
    """
    # FIXME: Put d par in globals (load before function defs)
    # Create a forgetting matrix by which to multiply the increments
    F = np.power(1+cTime-T[~np.isnan(T)], -d)

    # The base is the sum of all increments, where each increment is reduced by
    # the forgetting function F
    B = np.multiply(S[~np.isnan(T)], F)
    B = np.nansum(B, axis=0)
    return(priorB+B)


def eval_links_sparse(S, T, cTime, priorB=0, d=0.2):
    """
    Calculates the current strength of all links, based on the
    time of occurences (T), the increments at each time (S), and the current
    time (cTime).

    Parameters
    -----------
    T : a 3d array where each row is a single time vector, and the 2nd and 3rd
    dimensions are a square stim matrix. A positive value means that two items
    where experienced together. Cells contain the timestamp of when the event
    S : similar to T in shape and size, but stores the size of the increment at
        each occurence
    cTime: a scalar for the current time
    priorB: prexisting strength. Default is zero, but you can supply
            a pandas series with the calculated prior bases
    """
    # FIXME: Put d par in globals (load before function defs)
    # Create a forgetting matrix by which to multiply the increments
    T1 = T.copy()
    T1.data -= 1+cTime
    T1.data[T1.data > 0] = 0
    T1.data = np.power(-T1.data, -d)
    T1.data[np.isinf(T1.data)] = 0

    # The base is the sum of all increments, where each increment is reduced by
    # the forgetting function F
#    S.data *= T.data
    B = S.multiply(T1)
    B = B.sum(axis=0)
    return(priorB+B)


def run_experiment(concepts, trials, exptype, encoding_prob=1):
    """ 
    Creates a separate network for each subject, populates it with concept nodes
    and runs simulation of the trials within a specific context node
    <exptype>: one of ['single-item-recognition','cued-recall']
    """
    nets = {}
    for subj in trials.subject.unique():
        if 'independent_session' in trials.columns:
            for session in trials[trials.subject == subj].independent_session.unique():
                net_key = subj+'_sess'+str(session)
                nets[net_key] = Network(net_key, encoding_prob)
                nets[net_key].populate(concepts)
                nets[net_key].run_trials(trials[(trials.subject == subj) & (trials.independent_session == session)], exptype)                
        else:
            nets[subj] = Network(subj, encoding_prob)
            nets[subj].populate(concepts)
            nets[subj].run_trials(trials[trials.subject == subj], exptype)
            
            
def eval_base_sparse(S, T, cTime, d, priorB=0, structure='nodes'):
    """
    Calculates the current base activation of all nodes/links, based on the
    time of occurences (T), the increments at each time (S), and the current
    time (cTime).

    Parameters
    -----------
    T : a matrix where each row is a single sitmulus vector, and each column
        a single trial vector. Cells contain the timestamp of when the event
        occured (if only one stimulus is displayed on a trial, it is a column
        with 0s for all other stimuli, and the timing for the presented stim)
    S : similar to T in shape and size, but stores the size of the increment at
        each occurence
    cTime: a scalar for the current time
    priorB: prexisting base activation. Default is zero, but you can supply
            a pandas series with the calculated prior base for concept nodes
    structure: string
        'links' or 'nodes'
    """
#    bp()
    # FIXME: Put d par in globals (load before function defs)
    # Create a forgetting matrix by which to multiply the increments
    T1 = -T.copy()
    T1.data += 1+cTime
    T1.data[T1.data < 0] = 0
    T1.data = np.power(T1.data, -d)
    T1.data[np.isinf(T1.data)] = 0

    # The base is the sum of all increments, where each increment is reduced by
    # the forgetting function F
#    S.data *= T.data
    B = S.multiply(T1)
    B = B.sum(axis=1)
    return(B)