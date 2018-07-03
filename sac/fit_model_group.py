import random
import scipy as sp
import numpy as np
import pandas as pd
from pdb import set_trace as bp

from sac.utils import profile, extract_results
import sac.utils

def rmse(obs, pred):
    return(np.nanmean((obs-pred) ** 2) ** 0.5)
    
@profile
def activation_to_response(par, x, other=0, zero_prob=False):
    """ Gets the area under a normal curve with mean par[0] and sd par[1] """
    resp = sp.stats.norm.cdf(x-par[0], loc=0, scale=par[1])
    # set probability to 0 if node doesn't exist
    if zero_prob:
        resp = [0 if x[0] == 0 else x[1] for x in zip(x, resp)]
    resp = np.array(resp)
    return (1-other) * resp 


class Model(object):
    ACT_THRESHOLD_BOUNDS = (0,5)
    ACT_SD_BOUNDS = (0.01,5)
    ACT_THRESHOLD_INIT = 1        
    ACT_SD_INIT = 1
        
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose, zero_prob=False, summarise='before'):
        self.nets = nets
        self.pars = init_pars
        self.par_names = par_names
        self.par_ranges = par_ranges
        self.equations = equations
        self.fit_filter = fit_filter
        self.group_vars = group_vars
        self.resp_vars = resp_vars
        self.verbose = verbose
        self.reject_related = reject_related
        self.obs_data = (target_obs.
                         query(fit_filter).
                         groupby(group_vars)
                         [self.resp_vars].
                         mean())
        self.target_obs = target_obs
        self.zero_prob = zero_prob
        self.summarise = summarise
        self.opt_pars = {}
        self.opt_grid = {} 
        
    def save_pars(self, name):
        fitted_pars = {**self.pars, **self.opt_pars}
        fitted_pars['study'] = name
        fitted_pars = pd.DataFrame([fitted_pars])
        fitted_pars.to_csv('output/final/'+name+'_fitted_pars.csv')
        
    @profile
    def estimate_parameters(self, order='joint', func_type='fmin', rep=1, ftol=0.001, **kwargs):
        self.rep = rep
        self.ftol = ftol
        
        funcs = {'sequential': {'brute': self.estimate_sequential_brute,},
#                                'fmin':  self.estimate_sequential_fmin},
                 'joint': {'fmin':  self.estimate_joint_fmin,
                           'brute': self.estimate_joint_brute}}
        if self.par_names is not None: funcs[order][func_type]()
        self.fit_with_final_parameters()   
        
    @profile    
    def estimate_sequential_brute(self):
        for par_name in self.par_names:
            print("\n#######################################\n#")
            print("# Fitting Par: %s" % par_name)
            print("#\n#######################################\n")     
            self.c_par_name = [par_name]
            par_range = (self.par_ranges[par_name],)
            
            op = sp.optimize.brute(self.error_function, par_range, 
                                   full_output=1, finish=None, 
                                   disp=self.verbose)
            
            self.opt_pars[par_name] = op[0]
            self.pars[par_name] = op[0]
            if self.verbose: print(self.opt_pars) 
            
    @profile     
    def estimate_joint_brute(self):     
        self.c_par_name = self.par_names
        par_ranges = [[]] * len(self.par_names)
        for i, par_name in enumerate(self.par_names):
            par_ranges[i] = (self.par_ranges[par_name])
        par_ranges = tuple(par_ranges)
            
        op = sp.optimize.brute(self.error_function, par_ranges, 
                               full_output=1, finish=None, 
                               disp=self.verbose)
        
        for par_name, par in zip(self.par_names, op[0]):    
            self.opt_pars[par_name] = par
            self.pars[par_name] = par
        if self.verbose: print(self.opt_pars)            
        self.opt_grid = op          
            
    @profile        
    def estimate_joint_fmin(self):
        print("\n#######################################\n#")
        print("# Fitting Group")
        print("#\n#######################################\n")
        self.c_par_name = self.par_names     
        op = [[]] * self.rep
        x0 = np.zeros(len(self.par_names))
        for rep in range(self.rep):              
            for i, par_name in enumerate(self.par_names):
                x = self.par_ranges[par_name]
                x = np.arange(x.start, x.stop, x.step)
                x0[i] = random.choice(x)
                if self.verbose: print('# Init value for %s: %f' % (par_name, x0[i]))
            
            print("# REP: %d" % rep)
            op[rep] = sp.optimize.fmin(self.error_function, 
                                      x0, xtol=min(x0/100), ftol=self.ftol,
                                      full_output=1, maxfun=500, 
                                      disp=self.verbose)
        op = sorted(op, key=lambda x: x[1])[0]
            
        for par_name, par in zip(self.par_names, op[0]): 
            self.opt_pars[par_name] = par
            self.pars[par_name] = par
        if self.verbose: print(self.opt_pars)               
            
    @profile
    def error_function(self, pars):
        # brute opt returns a 0d arrays, so fix that
        pars = pars.flatten()
        pars_dict = self.pars.copy()
        for par_name, par in zip(self.c_par_name, pars):
            if self.verbose: 
                print('\n### Fitting parameter: "%s" with value %f\n' % 
                      (par_name, par))
            pars_dict[par_name] = par
        opt = self.run_nets_and_fit_activation_pars(pars_dict, act_col='epiA')
        error = 0
        for op in opt.values():
            error += op[1]
        return(error) 
        
    @profile        
    def run_nets_and_fit_activation_pars(self, pars, act_col):
        for net in self.nets.values():
#            stim_info = sac.utils.get_stim_info(net.trials, list(self.nets.values())[0].stimColumns, pars)
#            net.extract_prior_base(stim_info)            
            net.run_trials(pars, self.equations)
        self.results = extract_results(self.nets)
        # for free recall, simulate output interference
        if 'testtype' in self.results.columns:
            if any(self.results['testtype'] == 'free_recall') and 'free_recall_exp' in pars.keys():
                self.results['epiA'][self.results['testtype'] == 'free_recall'] = np.exp(pars['free_recall_exp']*self.results['epiA'][self.results['testtype'] == 'free_recall'])/10000
        if self.fit_filter is not None:
            self.results_full = self.results.copy()
            self.results = self.results.query(self.fit_filter)           
        opt = self.fit_activation_to_response(act_col, self.resp_vars[0])
        return({act_col: opt})

    @profile
    def fit_activation_to_response(self, act_col, resp_var, other=0):
        self.act_smry = self.results.groupby(self.group_vars)[act_col].mean()
        self.obs_data_smry = self.obs_data.loc[self.act_smry.index][resp_var]
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                               args=(act_col,
                                     resp_var,
                                     other), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
        self.results[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                                           x = self.results[act_col].values, 
                                           other = other, zero_prob = self.zero_prob)
        self.pred_smry = self.act_smry.reset_index()
        self.pred_smry[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                                           x = self.act_smry, 
                                           other = other, zero_prob = self.zero_prob)
        return(opt)      

    @profile
    def activation_error(self, par, act_col, resp_var, other):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        # @TODO: Find a way to do this more efficiently
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs
        if within_bounds:   
            if self.summarise == 'before':
                pred_data = self.act_smry
                resp_pred = activation_to_response(par, pred_data.values, other, self.zero_prob)
            if self.summarise == 'after':
                pred_data = self.results[self.group_vars+[act_col]]
                resp_pred = activation_to_response(par, pred_data[act_col].values, other, self.zero_prob)
                pred_data['pred'] = resp_pred
                resp_pred = pred_data.groupby(self.group_vars)['pred'].mean().values
            obs = self.obs_data_smry
            return rmse(obs.values, resp_pred)
        return 1   
    
    @profile
    def fit_with_final_parameters(self):
        pars = self.pars.copy()
        opt = self.run_nets_and_fit_activation_pars(pars, act_col='epiA')               
        for key, value in opt.items():
            self.opt_pars[key+'_act_theta'] = value[0][0]
            self.opt_pars[key+'_act_sd'] = value[0][1]
            self.opt_pars['error'] = value[1]
        if self.verbose: print(self.opt_pars)
        
        
class CoxModel(Model):
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       resp_vars, reject_related, verbose)
        
    @profile        
    def run_nets_and_fit_activation_pars(self, pars, act_col):
        for net in self.nets.values():
            net.run_trials(pars, self.equations)
        self.results = extract_results(self.nets)
        self.results['epiA'][self.results['condition'] == 'single recognition'] += self.results['epiA'][self.results['condition'] == 'single recognition']/1.5
        if self.fit_filter is not None:
            self.results_full = self.results.copy()
            self.results = self.results.query(self.fit_filter)           
        opt = self.fit_activation_to_response(act_col, self.resp_vars[0])
        return({act_col: opt})
        
class MarevicModel(Model):
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       resp_vars, reject_related, verbose)
        
        
    def fit_activation_to_response(self, act_col, resp_var, other=0):
        # fit lag effect
        fit = (self.results.melt(id_vars=['procedure','cue_type','epiA',resp_var], 
                    value_vars=['cue_type_prioritem1',
                                'cue_type_prioritem2',
                                'cue_type_prioritem3',
                                'cue_type_prioritem4']))
        fit = (fit.query('procedure == "test" & (value == "tbr" | value == "tbf")').
                   groupby(['variable','value','cue_type'])['epiA',resp_var].
                   mean())
        self.act_smry = fit['epiA']
        self.obs_data_smry = fit[resp_var]
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                               args=(act_col,
                                     resp_var,
                                     other), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
        self.results[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                                           x = self.results[act_col].values, 
                                           other = other, zero_prob = self.zero_prob)
        self.pred_smry = self.act_smry.reset_index()
        self.pred_smry[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                                           x = self.act_smry, 
                                           other = other, zero_prob = self.zero_prob)
        return(opt)      
        
class OldNewItemRecognitionModel(Model):
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose, zero_prob, summarise):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       resp_vars, reject_related, verbose, zero_prob, summarise)
             
    def run_nets_and_fit_activation_pars(self, pars, act_col):
        for net in self.nets.values():
            net.run_trials(pars, self.equations)
        self.results = extract_results(self.nets)
        if self.fit_filter is not None:
            self.results_full = self.results.copy()
            self.results = self.results.query(self.fit_filter)              
        opt = self.fit_activation_to_response(['epiA','stim1.semB'], self.resp_vars[0]) 
        return({'rem': [opt[0][0:2],opt[1]], 'know': [opt[0][2:4],opt[1]]})
        
    def fit_activation_to_response(self, act_cols, resp_var, other=0):
        self.act_smry = self.results.groupby(self.group_vars)[act_cols].mean()
        self.obs_data_smry = self.obs_data.loc[self.act_smry.index][resp_var]
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT,
                                     self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                               args=(act_cols,
                                     resp_var,
                                     other), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
                               
        self.results['old_rem_pred'] = activation_to_response([opt[0][0],opt[0][1]], 
                                           self.results.epiA.values, 0,
                                           zero_prob = self.zero_prob)   
        self.results['old_know_pred'] = activation_to_response([opt[0][2],opt[0][3]], 
                                           self.results['stim1.semB'].values, 
                                           self.results['old_rem_pred'],
                                           zero_prob = self.zero_prob)                               
        self.results[resp_var+'_pred'] = self.results['old_rem_pred']+self.results['old_know_pred']
#        self.pred_smry = self.act_smry.reset_index()
#        self.pred_smry[resp_var+'_pred'] = activation_to_response(par = opt[0], 
#                                           x = self.act_smry, 
#                                           other = other, zero_prob = self.zero_prob)
        return(opt)    
        
    def activation_error(self, par, act_cols, resp_var, other):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        # @TODO: Find a way to do this more efficiently
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs
        if within_bounds:   
#            if self.summarise == 'before':
#                pred_data = self.act_smry
#                resp_pred = activation_to_response(par, pred_data.values, other, self.zero_prob)
            if self.summarise == 'after':
                pred_data = self.results[self.group_vars+act_cols]
                rem_pred = activation_to_response([par[0],par[1]], pred_data.epiA, 0, self.zero_prob) 
                know_pred = activation_to_response([par[2],par[3]], pred_data['stim1.semB'], rem_pred) 
                pred_data['rem_pred'] = rem_pred
                pred_data['know_pred'] = know_pred
                pred_data['pred'] = rem_pred+know_pred
                resp_pred = pred_data.groupby(self.group_vars)['pred'].mean().values
            obs = self.obs_data_smry
            self.pred_smry = pred_data.groupby(self.group_vars)['pred'].mean()
            return rmse(obs.values, resp_pred)
        return 1   
    
    
class OldNewItemRecognitionModelCombinedAct(Model):
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose, zero_prob, summarise):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       resp_vars, reject_related, verbose, zero_prob, summarise)
        
    def run_nets_and_fit_activation_pars(self, pars, act_col):
            for net in self.nets.values():
                net.run_trials(pars, self.equations)
            self.results = extract_results(self.nets)
            self.results['epiA'] = (1-pars['sem_weight'])*self.results['epiA'] + pars['sem_weight']*self.results['stim1.semB']
            if self.fit_filter is not None:
                self.results_full = self.results.copy()
                self.results = self.results.query(self.fit_filter)           
            opt = self.fit_activation_to_response(act_col, self.resp_vars[0])
            return({act_col: opt})        
             
    
    
class OldNewItemRecognitionModelFixedSd(Model):
    def __init__(self, nets, init_pars, par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 resp_vars, reject_related, verbose, zero_prob, summarise):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       resp_vars, reject_related, verbose, zero_prob, summarise)
             
    def run_nets_and_fit_activation_pars(self, pars, act_col):
        for net in self.nets.values():
            net.run_trials(pars, self.equations)
        self.results = extract_results(self.nets)
        if self.fit_filter is not None:
            self.results_full = self.results.copy()
            self.results = self.results.query(self.fit_filter)              
        opt = self.fit_activation_to_response(['epiA','stim1.semB'], self.resp_vars[0]) 
        return({'both': opt})
        
    def fit_activation_to_response(self, act_cols, resp_var, other=0):
        self.act_smry = self.results.groupby(self.group_vars)[act_cols].mean()
        self.obs_data_smry = self.obs_data.loc[self.act_smry.index][resp_var]
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT,
                                     self.ACT_THRESHOLD_INIT], 
                               args=(act_cols,
                                     resp_var,
                                     other), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
                               
        self.results['old_rem_pred'] = activation_to_response([opt[0][0],0.1], 
                                           self.results.epiA.values, 0,
                                           zero_prob = self.zero_prob)   
        self.results['old_know_pred'] = activation_to_response([opt[0][1],0.4], 
                                           self.results['stim1.semB'].values, 
                                           self.results['old_rem_pred'],
                                           zero_prob = self.zero_prob)                               
        self.results[resp_var+'_pred'] = self.results['old_rem_pred']+self.results['old_know_pred']
#        self.pred_smry = self.act_smry.reset_index()
#        self.pred_smry[resp_var+'_pred'] = activation_to_response(par = opt[0], 
#                                           x = self.act_smry, 
#                                           other = other, zero_prob = self.zero_prob)
        return(opt)    
        
    def activation_error(self, par, act_cols, resp_var, other):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        # @TODO: Find a way to do this more efficiently
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        within_bounds = par[0] > lt and par[1] > lt and par[0] < ht and par[1] < ht
        if within_bounds:   
#            if self.summarise == 'before':
#                pred_data = self.act_smry
#                resp_pred = activation_to_response(par, pred_data.values, other, self.zero_prob)
            if self.summarise == 'after':
                pred_data = self.results[self.group_vars+act_cols]
                rem_pred = activation_to_response([par[0],0.1], pred_data.epiA, 0, self.zero_prob) 
                know_pred = activation_to_response([par[1],0.4], pred_data['stim1.semB'], rem_pred) 
                pred_data['rem_pred'] = rem_pred
                pred_data['know_pred'] = know_pred
                pred_data['pred'] = rem_pred+know_pred
                resp_pred = pred_data.groupby(self.group_vars)['pred'].mean().values
            obs = self.obs_data_smry
            self.pred_smry = pred_data.groupby(self.group_vars)['pred'].mean()
            return rmse(obs.values, resp_pred)
        return 1  
    
    def fit_with_final_parameters(self):
        pars = self.pars.copy()
        opt = self.run_nets_and_fit_activation_pars(pars, act_col='epiA')               
#        for key, value in opt.items():
#            self.opt_pars[key+'_act_theta'] = value[0][0]
#            self.opt_pars[key+'_act_sd'] = value[0][1]
#            self.opt_pars['error'] = value[1]
#        if self.verbose: print(self.opt_pars)
        
        
        
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import scipy as sp
    import os
    import matplotlib.pyplot as plt
    import random
    
    import warnings 
    warnings.simplefilter("ignore")
    from plotnine import *
    
    import sac.main
    import sac.utils
    import sac.fit_model 
    
    from importlib import reload
    reload(sac.equations)
    reload(sac.main)
    reload(sac.utils)
    reload(sac.fit_model)
    
    os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')
    
    subjects_to_fit = [1,2,3]
    #subjects_to_fit = random.samp
    
    """ DEFINE CONSTANTS TO USE IN THE MODEL """
    init_pars = {'SCALE': 1000.0,
                 'W': 3,
                 'contextAct': 1,
                 'decaytype': 'power',
                 'dl': -0.12,
                 'dn': -0.18,
                 'fan': 2,
                 'p': 0.8,
                 'p_forget_prop': 0.95,
                 'w_act': 2,
                 'prior0': 0.4,
                 'prior1': 0.2,
                 'prior2': -0.1,
                 'sem_theta': 1,
                 'w_recovery_rate': 0.9,
                 'y': 0.2}
    
    
    """ SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
    equations = {'conceptLearning': 'learning_equation10',
                 'episodeLearning': 'learning_equation10',
                 'linkLearning': 'learning_equation2',
                 'conceptDecay': 'decay_power',
                 'episodeDecay': 'decay_power',
                 'linkDecay': 'decay_power'}
        
    filter_cond = ('procedure == "test" ' +
                  'and cue_type_prioritem1 != "nan"')
    
    stim_columns = ['stim1','stim2','list'] 
    group_vars = ['cue_type','cue_type_consec_value'] 
    split_nets_by = ['subject']
    
    """ LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
    allsubjects = sac.utils.load_trials('data/marevic2017_exp1.csv', 
                                        scale=init_pars['SCALE'])
    trials = sac.utils.select_subjects(allsubjects, subjects_to_fit)
    stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
    stim_info['priorBase'] = 0.3
    stim_info['priorFan'] = 0.6
    nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                                   init_pars, stim_info, duration=True)
    
    
    model = Model(nets=nets, 
                            init_pars=init_pars, 
                            par_names=['w_recovery_rate','p_forget_prop'],
                            par_ranges={'w_recovery_rate': slice(0.6,0.7,0.15),
                                                      'W': slice(3,5,0.1),
                                                      'p': slice(0.2,0.99,0.05),
                                          'p_forget_prop': slice(0.1,0.15,0.1),
                            },              
                            equations=equations, 
                            group_vars=group_vars, 
                            target_obs=allsubjects, 
                            resp_vars=['cued_recall_acc'], 
                            fit_filter=filter_cond,
                            verbose=True,
                            reject_related=False)
    
    model.estimate_parameters(order='joint', func_type='brute', rep=1, ftol=0.001, duration=True)