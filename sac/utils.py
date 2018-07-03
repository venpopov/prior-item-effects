import pandas as pd
import nltk
import numpy as np
import scipy as sp
import scipy.sparse as sps
from pdb import set_trace as bp
import time
import itertools
import random
import os
import warnings
import builtins
import cProfile
pd.options.mode.chained_assignment = None
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        if kwargs.get('duration',False):
            print('%s function took %0.3f s' % (f.__name__, (time2-time1)))
        return ret
    return wrap

def profile(f):
    try:
        profile = builtins.profile
    except AttributeError:
        profile = timing
    return(profile(f))

def do_profile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func
    

def generate_words(n, minFreq, maxFreq, exclude=[''], freqFilePath=os.path.join(dir_path, 'SUBTLEX.txt'), seed=0):
    if seed != 0:
        np.random.seed(seed)
    wordFreq = pd.read_table(freqFilePath)
    wordFreq = wordFreq[['Word', 'SUBTLWF']]
    pool = wordFreq[(wordFreq['SUBTLWF'] >= minFreq) & (wordFreq['SUBTLWF'] <= maxFreq)]
    pool = pool[pool['Word'] == pool['Word'].str.lower()]
    pool = pool[~pool['Word'].isin(exclude)]
    pool = pool.reset_index(drop=True)
    idx = np.random.choice(pool.shape[0], size=n, replace=False)
    sample = pool.iloc[idx]
    sample.columns = ['word', 'freq']
    sample = sample.reset_index(drop=True)
    return(sample)


def load_trials(filepath, scale=1, lower=True, encoding='utf-8'):
    """
    Reads a csv file with data from the experiment.
    Adds columns in which to save the concept and event nodes activation values
    The following columns are required:

    # trial: trial number
    # stim: the name of the studied word/concept. Must match names in the
            frequency file
    # list: number of the studied list. A new context node will be created for 
            each unique entry
    # onset: the onset of the trial with respect to the begining of the
             experiment. The first trial should have onset 0.
             Assumed in seconds. If else, specify 'scale', such that
             onset/scale is in seconds
    """
    
    data = pd.read_csv(filepath, encoding=encoding)
    # make string columns contents lower case
    if lower:
        for column in data:
            if data[column].dtype == 'O':
                data[column] = data[column].astype(str).str.lower()
    data['onset'] = data['onset']/scale
    data['list'] = 'context' + data['list'].astype(str)
    return(data)

def get_word_freq(trials, stim_columns=['stim'], freqFilePath=os.path.join(dir_path, 'SUBTLEX.txt')):
    """ returns a dataframe with word frequency for the supplied dataframe """
    wordFreq = pd.read_table(freqFilePath)
    wordFreq = wordFreq[['Word', 'SUBTLWF','SUBTLCD']]
    words = []
    for col in stim_columns:
        words = words + list(trials[col].unique())
    words = list(set(words))
    words = [word.lower() for word in words]
    found = wordFreq.loc[wordFreq.Word.isin(words)]
    nonFound = list(filter(lambda x: x not in found.Word.tolist(), words))
    foundUpperCase = wordFreq.loc[wordFreq.Word.str.lower().isin(nonFound)]
    foundUpperCase.Word = foundUpperCase.Word.str.lower()
    found = found.append(foundUpperCase)
    found['exists'] = 'yes'
    nonFound = list(filter(lambda x: x not in found.Word.tolist(), words))
    lowestFreq = wordFreq.SUBTLWF.min()
    lowestFreq = [lowestFreq] * len(nonFound)
    lowestCD = wordFreq.SUBTLCD.min()
    lowestCD = [lowestCD] * len(nonFound)
    nonFound = pd.DataFrame({'Word': nonFound, 'SUBTLWF': lowestFreq, 'SUBTLCD': lowestCD, 'exists': 'no'})
    stimFreq = found.append(nonFound)
    stimFreq.index = stimFreq.Word
    stimFreq = stimFreq.rename(columns={'SUBTLWF': 'freq','SUBTLCD': 'cv'})
    stimFreq = stimFreq.sort_values('freq')
    return(stimFreq)


def generate_wordlist_from_trials(trials, stim_columns=['stim']):
    """
    generates a dictionary of words as keys and frequencies from the brown
    corpus as values. It uses the words in the trial file that are present in
    columns named in the <stim_columns> list
    """
    # TODO: get SUBTLEX frequency instead
    words = []
    for col in stim_columns:
        words = words + list(trials[col].unique())
    words = list(set(words))
    words = [word.lower() for word in words]
    brown_freq = nltk.FreqDist(nltk.corpus.brown.words())
    word_freq = {key: brown_freq[key] for key in words}
    word_freq1 = {k: v+1 for (k, v) in word_freq.items() if v == 0}
    word_freq2 = {k: v for (k, v) in word_freq.items() if v != 0}
    word_freq = word_freq1
    word_freq.update(word_freq2)

    # transform to pandas data.frame
    word_freq = pd.DataFrame.from_dict(word_freq, orient='index')
    word_freq = word_freq.sort_index()
    word_freq.columns = ['freq']

    return(word_freq)


def get_stim_info(trials, stim_columns, par, freqFilePath=os.path.join(dir_path, 'SUBTLEX.txt')):
    """
    Prepares a pandas dataframe that contains on each row info about each
    stimulus = word frequency, conditions, prior base levela activation
    """
    from sac.equations import calc_preexisting_base, calc_preexisting_fan
    # Get frequency information
    stimInfo = get_word_freq(trials, stim_columns=stim_columns,
                             freqFilePath=freqFilePath)
#    stimInfo.freq.loc['context1'] = 0
#    stimInfo.freq[stimInfo['exists'] == 'no'] = 0

    # Get prior base level activation depending on frequency
    # FIXME: estimate the preexisting fan
    stimInfo['priorBase'] = stimInfo['freq'].apply(calc_preexisting_base, args=([par]))
    stimInfo['priorFan'] = stimInfo['cv'].apply(calc_preexisting_fan, args=([par]))
    stimInfo['nodeType'] = np.where(stimInfo.index.str.contains('context'),
                                    'context', 'concept')
    stimInfo.priorBase[stimInfo.index.str.contains('context')] = 0
    stimInfo.priorFan[stimInfo.index.str.contains('context')] = 0
#    stimInfo.priorBase[stimInfo['exists'] == 'no'] = 0
#    stimInfo.priorFan[stimInfo['exists'] == 'no'] = 0

    # TODO: Get experiment specific information
#    stimInfo['nrep'] = trials.groupby(['stim']).stim.count()
#    frequnique = trials.groupby(['stim']).freq.unique()
#    stimInfo['freqcat'] = frequnique.apply(lambda x: x[0])
    stimInfo = stimInfo.sort_values(['Word'])
    
    # remove prior for nan values, which are just missing cells
    if sum(stimInfo.index == 'nan') > 0:
        stimInfo.loc['nan', 'priorBase'] = 0
        stimInfo.loc['nan', 'freq'] = 0
        stimInfo.loc['nan', 'priorFan'] = 0
    return(stimInfo)
    

def change_par(currentParameters, newParameters):
    for key, value in newParameters.items():
        currentParameters[key] = value
    return(currentParameters)

def expand_grid(data_dict):
   product = itertools.product(*data_dict.values())
   return pd.DataFrame.from_records(product, columns=data_dict.keys())
    
def array_to_sparse(M):
    """
    Makes a sparse matrix from a numpy array
    """
    warnings.simplefilter("ignore")
    idx = np.nonzero(M > 0)
    M = sps.coo_matrix((M[idx], idx), shape=M.shape)
    M = M.tocsr()
    return(M)
    
def transpose_csr_to_csc(csr):
    def tmp(*wargs, **kwargs): pass
    check = sps.csc_matrix.check_format
    sps.csc_matrix.check_format = tmp
    M, N = csr.shape
    mat = sps.csc_matrix((csr.data, csr.indices, csr.indptr), shape=(N, M))
    sps.csc_matrix.check_format = check
    return(mat)
    
def update_csr(M, newData, rowId, colIds):
    """
    Takes a sparse csr matrix, a vector of new data, a row number for the data
    and a vector of column ids corresponding to each value in data
    """
    def add_values_to_csr(oldValues, newValues, rowId):
        rowOld = oldValues[:rowpb]
        otherRow = oldValues[rowpb:]
        values = np.concatenate((rowOld,newValues,otherRow))
        return(values)
        
    # references indices to extract colids for this row    
    rowpa = M.indptr[rowId]
    rowpb = M.indptr[rowId+1]
    
    # replace values for which positions there is currently a value in the matrix
    sharedIdx = np.intersect1d(M.indices[rowpa:rowpb], colIds)
    sharedIdxIdx = np.in1d(colIds, sharedIdx)
    M.data[sharedIdx] = newData[sharedIdxIdx]
        
    # add indices and data for positions that are not currently in the matrix   
    M.indices = add_values_to_csr(M.indices, colIds[~sharedIdxIdx], rowId).astype(np.int32)
    M.data = add_values_to_csr(M.data, newData[~sharedIdxIdx], rowId)
    # fix the reference between row indices and column indices
    M.indptr[rowId+1:] += len(colIds)
    return(M)


def select_subjects(df, subjects):
    """ filter dataframe to run experimenton selected subjects only """
    if subjects is not None:
        return df.loc[df.subject.isin(subjects)]
    return df

def summarise_obs_and_predictions(results, allsubjects, filter_cond, group_vars, resp_vars, pred_vars):
    obs = (allsubjects.
        query(filter_cond).
        groupby(group_vars)
        [resp_vars].
        mean().
        reset_index().
        melt(id_vars=group_vars, value_vars=resp_vars))
    pred = (results.
        query(filter_cond).
        groupby(group_vars)
        [pred_vars].
        mean().
        reset_index().
        melt(id_vars=group_vars, value_vars=pred_vars))
    results = pd.concat([pred, obs], axis=0)
    return(results)

def combine_parameters(constants, par):
    pars = expand_grid(par)
    for key, value in constants.items():
        if key in pars.columns:
            raise ValueError('Parameter <%s> specified both as a constant and as a variable!' % key)
        if type(value) is list:
            for i, v in enumerate(value):
                pars[key+str(i)] = v
        else:
            pars[key] = value
    return(pars)
    

def get_unique_episodes(trials, stim_columns):
    # FIXME: distinguish between study and test trials
    data = trials[[col for col in trials.columns if col in stim_columns]]
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    return(data)

def shuffle_list(some_list):
    """
    shuffles a list so that none of the original items are in the same position
    useful for recombining pairs
    """
    randomized_list = some_list[:]
    while True:
        random.shuffle(randomized_list)
        nRepeats = sum([x==y for x,y in zip(some_list, randomized_list)])
        if nRepeats == 0:
            return randomized_list
        
def subset_dict(keysList, ddict):
    """
    Returns a subset of the dictionary with only keys in keysList
    """
    subset = {key: value for key, value in ddict.items() if key in keysList}
    return(subset)
    
@profile
def init_networks(trials, group_vars, stim_columns, pars, stimInfo, **kwargs):
    """
    Creates a separate network for each subject (or variables specified in 
    group_vars), populates it with concept nodes
    """
    from sac.main import Network
    nets = {}
    grouped = trials.groupby(group_vars)
    
    for name, group in grouped:
        group_trials = group.reset_index(drop=True)
        nets[name] = Network(group_trials,
                             stimInfo,
                             stim_columns)
    return(nets)

@timing
def run_experiment(nets, pars, equations, verbose=True, **kwargs):
    """
    Takes networks for each subject, paramters and equations and runs the experiment
    """
    for subj in nets.keys():
        if verbose: print('Runing: subject %s' % (str(subj)))
        nets[subj].run_trials(pars, equations)  


def extract_results(networks):
    """ given a list of networks that have been run, returns a merged dataframe
    with the results from all """
    results = pd.DataFrame()
    for net in networks.values():
        results = results.append(net.results)
    return(results)

def csr_row_indices(M):
    """ Returns the row indices for the data values in a csr sparse matrix """
    return(np.nonzero(np.diff(M.indptr)))


""" FUNCTIONS FOR MODEL FITTING """
def get_fit(dataPred, dataObs, groupVars=['freq', 'rep']):
    def predict(par, x):
        """ Gets the area under a normal curve with mean par[0] and sd par[1] """
    #    pred = 1 / (par[0] + np.exp((par[1] - x/par[2])))
        pred = sp.stats.norm(0, par[1]).cdf(x-par[0])
        pred = [0 if x[0] == 0 else x[1] for x in zip(x, pred)]
        return(pred)

    def get_pred(par, data, valueVar, groupVars, other=0):
        data = data.copy()
#        if valueVar == 'semB': bp()
        x = data.filter(regex=valueVar).sum(axis=1)
        data['pred'] = predict(par, x)
        pred = data.groupby(groupVars).pred.mean()
        pred = (1-other) * pred
        return(pred)
    
    def get_obs(data, valueVar, groupVars):
        obs = data.groupby(groupVars)[valueVar].apply(np.mean)
        return(obs)
    
    def wrapper4fmin(par, dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs):
        """
        A generic wrapper function, which takes parameters, data for predictions, 
        data for observations, function for making predictions, function for making 
        observations, arguments for the prediction function and arguments for the 
        observation function. The it optimizes the parameters by minimizing the 
        rmse function defined below
        """
        def rmse(par, dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs):    
            pred = fPred(par, dataPred, **fPredArgs)
            obs = fObs(dataObs, **fObsArgs)
            error = np.nanmean(np.array((obs-pred) ** 2) ** 0.5)
            return(error)
        
        optPar = sp.optimize.fmin(rmse, par, args=(dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs), maxfun=2000)
        return(optPar)
    
    if 'rep' not in dataObs.columns:
        dataObs['rep'] = 1
        dataPred['rep'] = 1
    
    remPar = wrapper4fmin([1, 1], dataPred, dataObs, get_pred, get_obs, 
                          {'valueVar': 'epiA', 
                           'groupVars': groupVars},
                           {'valueVar': 'rem', 
                            'groupVars': groupVars})
    remPred = get_pred(remPar, dataPred, 'epiA', groupVars)
    knowPar = wrapper4fmin([5, 1], dataPred, dataObs, get_pred, get_obs, 
                          {'valueVar': 'semB', 
                           'groupVars': groupVars,
                           'other': remPred},
                           {'valueVar': 'know', 
                            'groupVars': groupVars})
    obs = get_obs(dataObs, ['rem','know'], groupVars)
    obs = obs.reset_index().melt(id_vars=groupVars)
    obs['type'] = 'data'
    knowPred = get_pred(knowPar, dataPred, 'semB', groupVars, remPred)
    pred =  pd.concat([remPred,knowPred], axis=1)
    pred.columns = ['rem','know']
    pred = pred.reset_index().melt(id_vars=groupVars)
    pred['type'] = 'model'
    res = obs.append(pred)
    error = np.nanmean((np.array(obs['value']-pred['value']) ** 2) ** 0.5)
    
    # get full data predictions for each trial
    x = dataPred.filter(regex='epiA').sum(axis=1)
    dataPred['rempred'] = predict(remPar, x)
    x = dataPred.filter(regex='semB').sum(axis=1)
    dataPred['knowpred'] = predict(knowPar, x) 
    dataPred['knowpred'] = (1-dataPred['rempred']) * dataPred['knowpred']
    
    fit = {'res': res, 'error': error, 'data': dataPred}
    return(fit)

def par_estimation_sequential(dataFit, dataObs, stimColumns, initPars, pars, equations, rejectInsertions, reps=2, testType='itemRecognition', groupVars=['freq','rep'], filtered=False):
    """
    - set which things are paramters
    - give them range values
    - loop over them, then consider only one as a parameter, the others as constants
    - fit the current parameter with grid search, then fix it to the best fitting value
    - continue with all and repeat twice
    """
    fitspars = []
    fittedPars = initPars.copy()
    for rep in range(reps):
        for par, values in pars.items():
            # current parameter to fit
            print('### Fitting parameter: %s\n' % par)
            currentPar = {par: np.unique(np.append(values, initPars[par]))}
            # fix other parameters as constants
            constants = {k: v for k,v in fittedPars.items() if k != par}
            # run experiment with these results and extract parameters
            nets = run_experiment(dataFit, stimColumns, constants, currentPar, equations, rejectInsertions)
            results = extract_results(nets)
            
            if filtered:
                results = results.loc[results['procedure'] == 'test']
                results = results.loc[~ ((results['freqprioritem'] == 'nan') & (results['triatype'] == 'old'))]
            results = results.reset_index(drop=True)
            if testType == 'itemRecognition':
                error = results.groupby(list(currentPar.keys())).apply(lambda x: get_fit(dataPred=x, dataObs=dataObs, groupVars=groupVars)['error'])
            elif testType == 'associativeRecognition':
                error = results.groupby(list(currentPar.keys())).apply(lambda x: get_fit_associative(dataPred=x, dataObs=dataObs)['error'])
            fittedPars[par] = error.idxmin()
            print('### Optimal value for paramter %s: %f\n' % (par, fittedPars[par]))
        fitspars.append(fittedPars)
    return({'error': error, 'fittedPars': fitspars})


def get_fit_associative(dataPred, dataObs):
    def predict(par, x):
        """ Gets the area under a normal curve with mean par[0] and sd par[1] """
    #    pred = 1 / (par[0] + np.exp((par[1] - x/par[2])))
        pred = sp.stats.norm(0, par[1]).cdf(x-par[0])
        pred = [0 if x[0] == 0 else x[1] for x in zip(x, pred)]
        return(pred)

    def get_pred(par, data, valueVar, groupVars, other=0):
        data = data.copy()
        # select columns with relevant values and sum them for a sum familiarity
        x = data.filter(regex=valueVar).sum(axis=1)
        data['pred'] = predict(par, x)
        pred = data.groupby(groupVars).pred.mean()
        pred = (1-other) * pred
        return(pred)
    
    def merge_pred(par, data, epiVar, semVar, groupVars):
        """
        Independently gets predictions for rememember and know responses and returns
        the proportion of old predicted responses
        """
        rem = get_pred(par[0:2], data, epiVar, groupVars)
        know = get_pred(par[2:], data, semVar, groupVars, other=rem)
        pred = rem.reset_index()
        pred = pred.rename(columns = {'pred': 'rem'})
        pred['know'] = know.reset_index()['pred']
        pred['old'] = pred.apply(lambda x: x['rem']+x['know'] if x['type'] == 'old' else x['know'], axis=1)
        return(pred)
    
    def get_obs(data, valueVar, groupVars):
        obs = data.groupby(groupVars)[valueVar].apply(np.mean)
        return(obs)
    
    def extract_old(par, data, epiVar, semVar, groupVars):
        pred = merge_pred(par, data, epiVar, semVar, groupVars)
        return(pred['old'])
    
    def extract_obs(data, valueVar, groupVars):
        obs = get_obs(data, valueVar, groupVars)
        return(obs.values)
    
 
    def wrapper4fmin(par, dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs):
        """
        A generic wrapper function, which takes parameters, data for predictions, 
        data for observations, function for making predictions, function for making 
        observations, arguments for the prediction function and arguments for the 
        observation function. The it optimizes the parameters by minimizing the 
        rmse function defined below
        """
        def rmse(par, dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs):   
            pred = fPred(par, dataPred, **fPredArgs)
            obs = fObs(dataObs, **fObsArgs)
            error = np.mean(np.array((obs-pred) ** 2) ** 0.5)
            return(error)
        
        optPar = sp.optimize.fmin(rmse, par, args=(dataPred, dataObs, fPred, fObs, fPredArgs, fObsArgs), maxfun=2000)
        return(optPar)
    
    obs = get_obs(dataObs, ['probability'], ['freq_cat','type'])
    obs = obs.reset_index().melt(id_vars=['freq_cat','type'])
    obs['source'] = 'data'
    obs = obs.rename(columns = {'value': 'old'})
    
    fitPar = wrapper4fmin([1, 1, 5, 1], dataPred, dataObs, extract_old, extract_obs, 
                          {'epiVar': 'epiA', 
                           'semVar': 'semB',
                           'groupVars': ['freq_cat','type']},
                           {'valueVar': 'probability', 
                            'groupVars': ['freq_cat','type']})
    pred = merge_pred(fitPar, dataPred, 'epiA','semB', ['freq_cat','type'])
    pred['source'] = 'model'
    res = obs.append(pred)
    error = np.mean((np.array(obs['old']-pred['old']) ** 2) ** 0.5)
    fit = {'res': res, 'error': error}
    return(fit)

def get_size_attr(obj):
    attr = obj.__dir__()
    d = {}
    for att in attr:
        x = obj.__getattribute__(att)
        d[att] = sys.getsizeof(x)
    d = pd.Series(d)
    d = d.sort_values()
    return(d)