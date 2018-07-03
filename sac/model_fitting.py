import scipy as sp
import numpy as np
import pandas as pd
import sac.utils
from sac.utils import profile
from pdb import set_trace as bp

@profile
def estimate_paramters_sequential(nets, initPars, commonPars, subjectPars, 
                                  parRanges, groupVars, equations, response_type, 
                                  filterQuery=None, finish=sp.optimize.fmin, 
                                  targetData=None, verbose=True, **kwargs):
    """
    Estimates paramters one at a time, while keeping all other fixed. 
        initPars = Dict include all paramter values, including constants
        commonPars = list of paramter names that should be fitted the same for all subjects
        subjectPars = list of parameter names that should be fit separately for each subject
        response_type = 'rem_know','old_new'...
        reps = how many cycles through all parameters
    """

    @profile
    def errorFunctionSubject(par):
        print('\n### Fitting parameter: "%s" with value %f\n' % (par_name, par))
        par = par.flatten()[0]
        pars = constants.copy()
        pars[par_name] = par
        actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, target_obs, filterQuery, verbose)
        error = []
        for i in range(len(pred)):
            error.append(rmse(obs[i],pred[i]))
        error = np.mean(error)
        return(error)
    
    def errorFunctionGroup(par):
        print('\n### Fitting parameter: "%s" with value %f\n' % (par_name, par))
        error = []
        for subject in subjects:
            net = nets[subject]
            subjID = np.unique(net.trials.subject)
            target_obs = targetData[subjID]
            pars = constants.copy()
            pars[par_name] = par            
            subjPars = optParsDict[subject]
            for key, value in subjPars.items():
                pars[key] = value
            actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, target_obs, filterQuery,verbose)
            for i in range(len(pred)):
                error.append(rmse(obs[i],pred[i]))
        error = np.mean(error)
        return(error)
    
    subjects = list(nets.keys())
    optPars = pd.DataFrame({'subject': subjects})
    optPars = optPars.set_index('subject') 
    
    # fit each subject paramter
    for subject in subjects:
        print("\n#######################################\n#")
        print("# Fitting Subject: %s" % str(subject))
        print("#\n#######################################\n")
        net = nets[subject]
        constants = initPars.copy()
        subjID = np.unique(net.trials.subject)
        target_obs = targetData.loc[subjID]
        target_obs.index = target_obs.index.droplevel('subject')
        for par_name in subjectPars:
            parRange = (parRanges[par_name],)
            # fit and save parameter
            output = sp.optimize.brute(errorFunctionSubject, parRange, full_output=1, finish=finish, disp=verbose)
            optPars.loc[subject, par_name] = output[0]
            optPars.loc[subject, 'error'] = output[1]
            constants[par_name] =  output[0]
        optPars.to_csv('tmp.csv')
        print(optPars)     

    # fit sample parameters
    optParsDict = optPars.to_dict('index')
    constants = initPars.copy()
    if commonPars is not None:
        for par_name in commonPars:
            print("\n#######################################\n#")
            print("# Fitting Group Par: %s" % par_name)
            print("#\n#######################################\n")
            parRange = (parRanges[par_name],)
            output = sp.optimize.brute(errorFunctionGroup, parRange, full_output=1, finish=finish, disp=verbose)
            optPars[par_name] = output[0]
            constants[par_name] = output[0]
            
    optParsDict = optPars.to_dict('index')


    print("\n#######################################\n#")
    print("# Get activation-response parameters")
    print("#\n#######################################\n")            
    for subject in subjects:
        pars = initPars.copy()
        net = nets[subject]
        subjPars = optParsDict[subject]
        for key, value in subjPars.items():
            pars[key] = value
        print(pars)
        actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, target_obs, filterQuery, verbose)
        for key, value in actPar.items():
            optPars.loc[subject, key+'_theta'] = value[0]
            optPars.loc[subject, key+'_sd'] = value[1]
        error = []
        for i in range(len(pred)):
            error.append(rmse(obs[i],pred[i]))
        error = np.mean(error)  
        optPars.loc[subject, 'error'] = error
            
    return(optPars)

def estimate_paramters_sequential_delete(trials, initPars, commonPars, subjectPars, parRanges, groupVars, equations, response_type, filterQuery=None, finish=sp.optimize.fmin, targetData=None, freqCorrection=False, verbose=True):
    """
    Estimates paramters one at a time, while keeping all other fixed. 
        initPars = Dict include all paramter values, including constants
        commonPars = list of paramter names that should be fitted the same for all subjects
        subjectPars = list of parameter names that should be fit separately for each subject
        response_type = 'rem_know','old_new'...
        reps = how many cycles through all parameters
    """

    def errorFunctionSubject(par):
        print('\n### Fitting parameter: "%s" with value %f\n' % (par_name, par))
        par = par.flatten()[0]
        pars = constants.copy()
        pars[par_name] = par
        actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, filterQuery, targetData)
        error = []
        for i in range(len(pred)):
            error.append(rmse(obs[i],pred[i]))
        error = np.mean(error)
        return(error)
    
    def errorFunctionGroup(par):
        print('\n### Fitting parameter: "%s" with value %f\n' % (par_name, par))
        smry = pd.DataFrame() 
        error = []
        for subject in subjects:
            nets = sac.utils.init_networks(trials.loc[trials['subject'] == subject], ['stim1','stim2','list'], initPars)
            net = nets[subject]
            if freqCorrection:
                net.stimInfo['priorBase'] = 0.30
                net.stimInfo['priorFan'] = 0.6
                net.cPriorB = net.cPriorB * 1.5
                net.cPriorFan = net.cPriorFan * 1.5
                net.concepts.Bprior = net.concepts.Bprior * 1.5
            pars = constants.copy()
            pars[par_name] = par            
            subjPars = optParsDict[subject]
            for key, value in subjPars.items():
                pars[key] = value
            actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, filterQuery, targetData)
            for i in range(len(pred)):
                error.append(rmse(obs[i],pred[i]))
            del net
        error = np.mean(error)
        return(error)
    
    subjects = trials.subject.unique()
    optPars = pd.DataFrame({'subject': subjects})
    optPars = optPars.set_index('subject')    
    
    # fit each subject paramter
    if subjectPars is not None:
        for subject in subjects:
            print("\n#######################################\n#")
            print("# Fitting Subject: %d" % subject)
            print("#\n#######################################\n")
            constants = initPars.copy()
            for par_name in subjectPars:
                nets = sac.utils.init_networks(trials.loc[trials['subject'] == subject], ['stim1','stim2','list'], initPars)
                net = nets[subject]
                if freqCorrection:
                    net.stimInfo['priorBase'] = 0.30
                    net.stimInfo['priorFan'] = 0.6
                    net.cPriorB = net.cPriorB * 1.5
                    net.cPriorFan = net.cPriorFan * 1.5
                    net.concepts.Bprior = net.concepts.Bprior * 1.5
                parRange = (parRanges[par_name],)
                
                # fit and save parameter
                output = sp.optimize.brute(errorFunctionSubject, parRange, full_output=1, finish=finish)
                optPars.loc[subject, par_name] = output[0]
                optPars.loc[subject, 'error'] = output[1]
                optPars.to_csv('marevic_subjects_fit.csv')
                constants[par_name] =  output[0]
            del net
            print(optPars)     

    # fit sample parameters
    optParsDict = optPars.to_dict('index')
    constants = initPars.copy()
    if commonPars is not None:
        for par_name in commonPars:
            print("\n#######################################\n#")
            print("# Fitting Group Par: %s" % par_name)
            print("#\n#######################################\n")
            parRange = (parRanges[par_name],)
            output = sp.optimize.brute(errorFunctionGroup, parRange, full_output=1, finish=finish)
            optPars[par_name] = output[0]
            constants[par_name] = output[0]
            
    optParsDict = optPars.to_dict('index')


    print("\n#######################################\n#")
    print("# Get activation-response parameters")
    print("#\n#######################################\n")            
    for subject in subjects:
        nets = sac.utils.init_networks(trials.loc[trials['subject'] == subject], ['stim1','stim2','list'], initPars)
        pars = initPars.copy()
        net = nets[subject]
        if freqCorrection:
            net.stimInfo['priorBase'] = 0.30
            net.stimInfo['priorFan'] = 0.6
            net.cPriorB = net.cPriorB * 1.5
            net.cPriorFan = net.cPriorFan * 1.5
            net.concepts.Bprior = net.concepts.Bprior * 1.5      
        subjPars = optParsDict[subject]
        for key, value in subjPars.items():
            pars[key] = value
        print(pars)
        actPar, fitted, pred, obs = run_and_fit(net, pars, equations, groupVars, response_type, filterQuery, targetData)
        for key, value in actPar.items():
            optPars.loc[subject, key+'_theta'] = value[0]
            optPars.loc[subject, key+'_sd'] = value[1]
        error = []
        for i in range(len(pred)):
            error.append(rmse(obs[i],pred[i]))
        error = np.mean(error)  
        optPars.loc[subject, 'error'] = error
        del net
            
    return(optPars)

@profile
def activation_error(par, bounds, data, obsCol, actCol, groupVars, target_obs, other=0):
    """ estimates the RMSE between observations and predictions, averaged over groupVars """
    error = 1
    if par[0]>bounds[0][0] and par[1]>bounds[1][0] and par[0]<bounds[0][1] and par[1]<bounds[0][1]:
        resp_pred = activation_to_response(par, data[actCol], other)
        data['pred'] = resp_pred
        obs = target_obs[obsCol]
        pred = data.groupby(groupVars)['pred'].mean()
        obs = obs.loc[pred.index]
        error = rmse(obs.values, pred.values)
    return(error)
    
@profile
def activation_to_response(par, x, other=0):
    """ Gets the area under a normal curve with mean par[0] and sd par[1] """
    x = x.values
    pred = sp.stats.norm.cdf(x-par[0], loc=0, scale=par[1])
    # set probability to 0 if no activation at all
    pred = [0 if x[0] == 0 else x[1] for x in zip(x, pred)]
    pred = (1-other) * pred
    return(pred)

def rmse(obs, pred):
    return(np.nanmean((obs-pred) ** 2) ** 0.5)
	
@profile
def run_and_fit(net, pars, equations, groupVars, response_type, target_obs, filterQuery=None, verbose=True):
    net.run_trials(pars, equations)
    results = net.results
    # if a filter is specified, filter the dataframe for trials only relevant for fitting   
    if filterQuery is not None:
        results = results.query(filterQuery)
        
    actPar, fitted = fit_response(results, groupVars, response_type, target_obs, verbose)
    # if fitting to a different dataset:

    if response_type == "rem_know":
        pred = summarise_data(fitted, groupVars, ['rem','know'])
        obs = summarise_data(target_obs.reset_index(), groupVars, ['rem','know'])  
    if response_type == "accuracy":
        pred = summarise_data(fitted, groupVars, ['CuedRecall'])
        obs = summarise_data(target_obs.reset_index(), groupVars, ['CuedRecall'])  
    return(actPar, fitted,  pred, obs)
  
@profile
def fit_response(data, groupVars, response_type, target_obs, verbose, **kwargs):
    par = {}
    data = data.copy()
    if response_type == "rem_know":
        # fit remember parameter
        par['rem'] = sp.optimize.fmin(activation_error, [1,1], args=([[0,5], [0.05,5]], data, 'rem', 'epiA', groupVars, target_obs), maxfun=500, disp=verbose)
        data['rem'] = activation_to_response(par['rem'], data['epiA'])
        # fit know parameter
        data['semB'] = data.filter(regex='semB').sum(axis=1)
        par['know'] = sp.optimize.fmin(activation_error, [1,1], args=([[0,5], [0.05,5]], data, 'know', 'semB', groupVars, target_obs, data['rem']), maxfun=500, disp=verbose)
        data['know'] = activation_to_response(par['know'], data['semB'], data['rem'])
        
    if response_type == "accuracy":
        par['acc'] = sp.optimize.fmin(activation_error, [1,1], args=([[0,5], [0.05,5]], data, 'CuedRecall', 'epiA', groupVars, target_obs), maxfun=500, disp=verbose)
        data['CuedRecall_pred'] = activation_to_response(par['acc'], data['epiA'])
        data['CuedRecall'] = activation_to_response(par['acc'], data['epiA'])
    data = data.drop('pred', axis=1)
    return(par, data)    

@profile    
def summarise_data(data, groupVars, valueVars):
    pred = data.melt(id_vars=groupVars, value_vars=valueVars, var_name='resp_type')
    pred = pred.groupby(groupVars+['resp_type'])['value'].mean()
    return(pred)

@profile        
def fit_stats(smry, groupVars):
    x = (smry.
         groupby(groupVars).
         value.mean().
         unstack('variable'))
    rmsd = rmse(x.obs,x.pred)
    rmsd = np.round(rmsd, 3)
    r = sp.corrcoef(x.obs, x.pred)[0,1]
    r2 = r ** 2
    r2 = np.round(r2,3)
    return(pd.DataFrame(dict(rmsd = [rmsd], r2 = [r2])))
    
    
if __name__ == "__main__":
    import os
    os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')
    from models.testing.test_marevic import *