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
def activation_to_response(par, x, other=0):
    """ Gets the area under a normal curve with mean par[0] and sd par[1] """
    resp = sp.stats.norm.cdf(x-par[0], loc=0, scale=par[1])
    # set probability to 0 if node doesn't exist
    resp = [0 if x[0] == 0 else x[1] for x in zip(x, resp)]
    resp = np.array(resp)
    return (1-other) * resp 


class Model(object):
    ACT_THRESHOLD_BOUNDS = (0,5)
    ACT_SD_BOUNDS = (0.01,5)
    ACT_THRESHOLD_INIT = 1        
    ACT_SD_INIT = 1
        
    def __init__(self, nets, init_pars, shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter,
                 reject_related, verbose):
        self.nets = nets
        self.pars = init_pars
        self.shared_par_names = shared_par_names
        self.subject_par_names = subject_par_names
        self.par_ranges = par_ranges
        self.equations = equations
        self.fit_filter = fit_filter
        self.group_vars = group_vars
        self.verbose = verbose
        self.subjects = list(self.nets.keys())
        self.reject_related = reject_related
        self.obs_data = (target_obs.
                         query(fit_filter).
                         groupby(group_vars)
                         [self.resp_vars].
                         mean())
        self.target_obs = target_obs
            
    @profile        
    def run_net_and_fit_activation_pars(self, pars, act_col):
        self.net.extract_prior_base(self.stim_info)
        self.net.run_trials(pars, self.equations)
        self.filter_results()
        opt = self.fit_activation_to_response(act_col, self.resp_vars[0])
        return({act_col: opt})  

    @profile
    def filter_results(self):
        """ filter the results for trials only relevant for fitting """  
        if self.fit_filter is not None:
            self.net.results_full = self.net.results.copy()
            self.net.results = self.net.results.query(self.fit_filter)          
    
    @profile
    def fit_activation_to_response(self, act_col, resp_var, other=0):
        self.net.smry = self.net.results.groupby(self.group_vars)[act_col].mean()
        self.subj_obs_data = self.obs_data.loc[self.net.smry.index][resp_var]
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                               args=(act_col,
                                     resp_var,
                                     other), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
        self.net.results[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                               x = self.net.results[act_col].values, 
                               other = other)
        return opt
        
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
           
            pred_data = self.net.smry
            resp_pred = activation_to_response(par, pred_data.values, other)
            obs = self.subj_obs_data
            return rmse(obs.values, resp_pred)
        return 1
    
    @profile
    def estimate_parameters(self, order, func_type, rep=1, ftol=0.001, **kwargs):
        self.opt_pars = pd.DataFrame({'subject': self.subjects})
        self.opt_pars = self.opt_pars.set_index('subject')
        self.opt_grid = {}
        self.rep = rep
        self.ftol = ftol
        
        funcs = {'sequential': {'brute': {'subj': self.estimate_subject_pars_sequential_brute,
                                          'group': self.estimate_group_pars_sequential_brute},
                                'fmin':  {'subj': self.estimate_subject_pars_sequential_fmin,
                                          'group': self.estimate_group_pars_sequential_fmin}},
                 'joint': {'brute': {'subj': self.estimate_subject_pars_joint_brute,
                                     'group': self.estimate_group_pars_joint_brute},
                           'fmin':  {'subj': self.estimate_subject_pars_joint_fmin,
                                     'group': self.estimate_group_pars_joint_fmin}}}
        
        if self.subject_par_names is not None: funcs[order][func_type]['subj']()
        if self.shared_par_names is not None: funcs[order][func_type]['group']()
        self.fit_subjects_with_final_parameters()
    
    @profile
    def fit_subjects_with_final_parameters(self):
        for subject in self.subjects:
            self.opt_pars_dict = self.opt_pars.to_dict('index')
            pars = self.pars.copy()
            subjPars = self.opt_pars_dict[subject]
            for key, value in subjPars.items():
                pars[key] = value
                
            self.net = self.nets[subject]
            self.stim_info = sac.utils.get_stim_info(self.target_obs, self.net.stimColumns, pars)                
            opt = self.run_net_and_fit_activation_pars(pars, act_col='epiA')     
            
            for key, value in opt.items():
                self.opt_pars.loc[subject, key+'_act_theta'] = value[0][0]
                self.opt_pars.loc[subject, key+'_act_sd'] = value[0][1]
                self.opt_pars.loc[subject, 'error'] = value[1]
        if self.verbose: print(self.opt_pars)
    
    @profile
    def error_function_subject(self, pars):
        # brute opt returns a 0d arrays, so fix that
        pars = pars.flatten()
        pars_dict = self.subj_pars.copy()
        for par_name, par in zip(self.c_par_name, pars):
            if self.verbose: 
                print('\n### Fitting parameter: "%s" with value %f\n' % 
                      (par_name, par))
            pars_dict[par_name] = par
        opt = self.run_net_and_fit_activation_pars(pars_dict, act_col='epiA')
        error = 0
        for op in opt.values():
            error += op[1]
        return(error)        
      
    @profile
    def error_function_group(self, pars):
        # brute opt returns a 0d arrays, so fix that
        pars = pars.flatten()
        for par_name, par in zip(self.c_par_name, pars):
            if self.verbose: 
                print('\n### Fitting parameter: "%s" with value %f\n' % 
                      (par_name, par))
                
        error = []
        
        for subject in self.subjects:
            self.net = self.nets[subject]
            subjPars = self.opt_pars_dict[subject]
            pars_dict = self.pars.copy()

            for key, value in subjPars.items():
                pars_dict[key] = value            
            for par_name, par in zip(self.c_par_name, pars):                
                pars_dict[par_name] = par 
                
            self.stim_info = sac.utils.get_stim_info(self.target_obs, self.net.stimColumns, pars_dict)
            opt = self.run_net_and_fit_activation_pars(pars_dict, act_col='epiA')
            
        fitted = extract_results(self.nets)
        resp_vars = [var+'_pred' for var in self.resp_vars]
        pred = fitted.groupby(self.group_vars)[resp_vars].mean()
        obs = self.obs_data.loc[pred.index][self.resp_vars]
        error =  rmse(obs.values, pred.values)
        print(error)
        return(error)

    @profile
    def estimate_subject_pars_sequential_brute(self):
        for subject in self.subjects:
            print("\n#######################################\n#")
            print("# Fitting Subject: %s" % str(subject))
            print("#\n#######################################\n")
            
            self.net = self.nets[subject]
            self.subj_pars = self.pars.copy()
            
            for par_name in self.subject_par_names:
                self.c_par_name = [par_name]
                par_range = (self.par_ranges[par_name],)
                
                op = sp.optimize.brute(self.error_function_subject, par_range, 
                                       full_output=1, finish=None, 
                                       disp=self.verbose)
                
                self.opt_pars.loc[subject, par_name] = op[0]
                self.subj_pars[par_name] =  op[0]
            
            self.opt_pars.loc[subject, 'error'] = op[1]
            self.opt_pars.to_csv('tmp.csv')
            if self.verbose: print(self.opt_pars)
    
    @profile    
    def estimate_group_pars_sequential_brute(self):
        for par_name in self.shared_par_names:
            print("\n#######################################\n#")
            print("# Fitting Group Par: %s" % par_name)
            print("#\n#######################################\n")
            self.opt_pars_dict = self.opt_pars.to_dict('index')      
            self.c_par_name = [par_name]
            par_range = (self.par_ranges[par_name],)
            
            
            op = sp.optimize.brute(self.error_function_group, par_range, 
                                   full_output=1, finish=None, 
                                   disp=self.verbose)
            
            self.opt_pars[par_name] = op[0]
            self.pars[par_name] = op[0]
            if self.verbose: print(self.opt_pars)
    
    @profile        
    def estimate_subject_pars_joint_brute(self):
        for subject in self.subjects:
            print("\n#######################################\n#")
            print("# Fitting Subject: %s" % str(subject))
            print("#\n#######################################\n")
            
            self.net = self.nets[subject]
            self.subj_pars = self.pars.copy()
            self.c_par_name = self.subject_par_names
            
            par_ranges = [[]] * len(self.subject_par_names)
            for i, par_name in enumerate(self.subject_par_names):
                par_ranges[i] = (self.par_ranges[par_name])
            par_ranges = tuple(par_ranges)
            
            op = sp.optimize.brute(self.error_function_subject, par_ranges, 
                                   full_output=1, finish=None, 
                                   disp=self.verbose)
            for par_name, par in zip(self.subject_par_names, op[0]):
                self.opt_pars.loc[subject, par_name] = par
                self.subj_pars[par_name] =  par
            
            self.opt_pars.loc[subject, 'error'] = op[1]
            self.opt_pars.to_csv('tmp.csv')
            if self.verbose: print(self.opt_pars)
            self.opt_grid[subject] = op
    
    @profile     
    def estimate_group_pars_joint_brute(self):
        self.opt_pars_dict = self.opt_pars.to_dict('index')      
        self.c_par_name = self.shared_par_names
        par_ranges = [[]] * len(self.shared_par_names)
        for i, par_name in enumerate(self.shared_par_names):
            par_ranges[i] = (self.par_ranges[par_name])
        par_ranges = tuple(par_ranges)
            
        op = sp.optimize.brute(self.error_function_group, par_ranges, 
                               full_output=1, finish=None, 
                               disp=self.verbose)
        
        for par_name, par in zip(self.shared_par_names, op[0]):    
            self.opt_pars[par_name] = par
            self.pars[par_name] = par
        if self.verbose: print(self.opt_pars)            
        self.opt_grid = op
        
    @profile
    def estimate_subject_pars_sequential_fmin(self):
        for subject in self.subjects:
            print("\n#######################################\n#")
            print("# Fitting Subject: %s" % str(subject))
            print("#\n#######################################\n")
                  
            self.net = self.nets[subject]
            self.subj_pars = self.pars.copy()
            
            for par_name in self.subject_par_names:
                self.c_par_name = [par_name]
                op = [[]] * self.rep
                for rep in range(self.rep):
                    x0 = self.par_ranges[par_name]
                    x0 = np.arange(x0.start, x0.stop, x0.step)
                    x0 = random.choice(x0)
                    if self.verbose: print('\n# Init value for %s: %f' % (par_name, x0))
                    
                    op[rep] = sp.optimize.fmin(self.error_function_subject, 
                                              x0,
                                              xtol=0.001, ftol=self.ftol,
                                              full_output=1, maxfun=500, 
                                              disp=self.verbose)
                op = sorted(op, key=lambda x: x[1])[0]
                
                self.opt_pars.loc[subject, par_name] = op[0]
                self.subj_pars[par_name] =  op[0]
            
            self.opt_pars.loc[subject, 'error'] = op[1]
            self.opt_pars.to_csv('tmp.csv')
            if self.verbose: print(self.opt_pars)
    
    @profile
    def estimate_group_pars_sequential_fmin(self):
        for par_name in self.shared_par_names:
            print("\n#######################################\n#")
            print("# Fitting Group Par: %s" % par_name)
            print("#\n#######################################\n")
            
            self.opt_pars_dict = self.opt_pars.to_dict('index')
            self.c_par_name = [par_name]
            op = [[]] * self.rep
            for i in range(self.rep):
                x0 = self.par_ranges[par_name]
                x0 = np.arange(x0.start, x0.stop, x0.step)
                x0 = random.choice(x0)                 
                op[i] = sp.optimize.fmin(self.error_function_group, 
                                          x0, xtol=x0/100, ftol=self.ftol,
                                          full_output=1, maxfun=500, 
                                          disp=self.verbose)
            op = sorted(op, key=lambda x: x[1])[0]
            self.opt_pars[par_name] = op[0][0]
            self.pars[par_name] = op[0][0]
            if self.verbose: print(self.opt_pars)
     
    @profile
    def estimate_subject_pars_joint_fmin(self):
        for subject in self.subjects:
            print("\n#######################################\n#")
            print("# Fitting Subject: %s" % str(subject))
            print("#\n#######################################\n")
                  
            self.net = self.nets[subject]
            self.subj_pars = self.pars.copy()
            self.c_par_name = self.subject_par_names
            x0 = np.zeros(len(self.subject_par_names))
            op = [[]] * self.rep
            for rep in range(self.rep):
                for i, par_name in enumerate(self.subject_par_names):
                    x = self.par_ranges[par_name]
                    x = np.arange(x.start, x.stop, x.step)
                    x0[i] = random.choice(x)
                    if self.verbose: print('\n# Init value for %s: %f' % (par_name, x0[i]))
                    
                op[rep] = sp.optimize.fmin(self.error_function_subject, 
                                          x0, xtol=min(x0/100), ftol=self.ftol,
                                          full_output=1, maxfun=500, 
                                          disp=self.verbose)
            op = sorted(op, key=lambda x: x[1])[0]
            
            for par_name, par in zip(self.subject_par_names, op[0]):    
                self.opt_pars.loc[subject, par_name] = par
            
            self.opt_pars.loc[subject, 'error'] = op[1]
            self.opt_pars.to_csv('tmp.csv')
            if self.verbose: print(self.opt_pars)
    
    @profile        
    def estimate_group_pars_joint_fmin(self):
        print("\n#######################################\n#")
        print("# Fitting Group")
        print("#\n#######################################\n")
        self.opt_pars_dict = self.opt_pars.to_dict('index')
        self.c_par_name = self.shared_par_names     
        op = [[]] * self.rep
        x0 = np.zeros(len(self.shared_par_names))
        for rep in range(self.rep):              
            for i, par_name in enumerate(self.shared_par_names):
                x = self.par_ranges[par_name]
                x = np.arange(x.start, x.stop, x.step)
                x0[i] = random.choice(x)
                if self.verbose: print('\n# Init value for %s: %f' % (par_name, x0[i]))
            
            print("# REP: %d" % rep)
            op[rep] = sp.optimize.fmin(self.error_function_group, 
                                      x0, xtol=min(x0/100), ftol=self.ftol,
                                      full_output=1, maxfun=500, 
                                      disp=self.verbose)
        op = sorted(op, key=lambda x: x[1])[0]
            
        for par_name, par in zip(self.shared_par_names, op[0]): 
            self.opt_pars[par_name] = par
            self.pars[par_name] = par
        if self.verbose: print(self.opt_pars)            

 
    @profile    
    def test(self, **kwargs):
        self.ap = []
        self.pred = []
        for i in self.nets.keys():
            self.net = self.nets[i]
            ap1 = self.run_net_and_fit_activation_pars(self.pars, 
                                                            act_col='epiA')
            self.ap.append(ap1)


class CuedRecallModel(Model):
    def __init__(self, nets, init_pars, shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, resp_vars, 
                 fit_filter=None, reject_related=False, verbose=True):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, shared_par_names, subject_par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       reject_related, verbose)

class AssociativeRecogModel(Model):
    def __init__(self, nets, init_pars, shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, resp_vars, 
                 fit_filter=None, reject_related=False, verbose=True):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, shared_par_names, subject_par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       reject_related, verbose)
    
    @profile        
    def run_net_and_fit_activation_pars(self, pars, act_col):
        self.net.run_trials(pars, self.equations)
        self.filter_results()
        self.net.results['stim.semB'] = (self.net.results['stim1.semB']+self.net.results['stim2.semB'])/2
        opt = self.fit_activation_to_response(['epiA','stim.semB'], self.resp_vars[0]) 
        return({'rem': [opt[0][0:2],opt[1]], 'know': [opt[0][2:4],opt[1]]})
    
    @profile    
    def fit_activation_to_response(self, act_cols, resp_var):
            self.net.smry = self.net.results.groupby(self.group_vars+['sameNode'])[act_cols].mean()
            self.subj_obs_data = self.obs_data.loc[self.net.smry.index.droplevel('sameNode')][resp_var]
            self.net.smry = pd.DataFrame(self.net.smry)
            opt = sp.optimize.fmin(func = self.activation_error, 
                                   x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT,
                                         self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                                   args=(act_cols,
                                         resp_var), 
                                   maxfun=500, full_output=True, disp=self.verbose,
                                   ftol = 0.001, xtol=0.001)
            # save results with fitted parameters
            resp_pred = activation_to_response([opt[0][0],opt[0][1]], self.net.results.epiA, 0) 
            self.net.results['old_rem_pred'] = resp_pred * self.net.results['sameNode'].values
            self.net.results['recomb_rem_pred'] = resp_pred * (1-self.net.results['sameNode'].values)
            epiA_resp = self.net.results['old_rem_pred']+self.net.results['recomb_rem_pred']
            # predictions based on semB
            resp_pred = activation_to_response([opt[0][2],opt[0][3]], self.net.results['stim.semB'], epiA_resp) 
            self.net.results['old_know_pred'] = resp_pred
            self.net.results['recomb_know_pred'] = (1-epiA_resp -resp_pred)
            #combined predictions 
            self.net.results[resp_var+'_pred'] = self.net.results['old_rem_pred']+self.net.results['old_know_pred']             
            return opt

    def activation_error(self, par, act_cols, resp_var):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        pred_data = self.net.smry[act_cols]
        obs = self.subj_obs_data
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs and par[2] > lt and par[3] > ls and par[2] < ht and par[3] < hs
        if within_bounds:
            # predictions based on epiA
            resp_pred = activation_to_response([par[0],par[1]], pred_data.epiA, 0) 
            self.net.smry['old_rem_pred'] = resp_pred * self.net.smry.reset_index(['sameNode'])['sameNode'].values
            self.net.smry['recomb_rem_pred'] = resp_pred * (1-self.net.smry.reset_index(['sameNode'])['sameNode'].values)
            epiA_resp = self.net.smry['old_rem_pred']+self.net.smry['recomb_rem_pred']
            # predictions based on semB
            resp_pred = activation_to_response([par[2],par[3]], pred_data['stim.semB'], epiA_resp) 
            self.net.smry['old_know_pred'] = resp_pred
            self.net.smry['recomb_know_pred'] = (1-epiA_resp -resp_pred)
            #combined predictions 
            resp_pred = self.net.smry['old_rem_pred']+self.net.smry['old_know_pred']
            return rmse(obs.values, resp_pred)
        return 1             
        
class OldNewItemRecognitionModel(Model):
    def __init__(self, nets, init_pars, shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, resp_vars, 
                 fit_filter=None, reject_related=False, verbose=True):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, shared_par_names, subject_par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       reject_related, verbose)
        
    def run_net_and_fit_activation_pars(self, pars, act_col):
        self.net.extract_prior_base(self.stim_info)
        self.net.run_trials(pars, self.equations)
        self.filter_results()
        opt = self.fit_activation_to_response(['epiA','stim1.semB'], self.resp_vars[0]) 
        return({'rem': [opt[0][0:2],opt[1]], 'know': [opt[0][2:4],opt[1]]})
        
    def fit_activation_to_response(self, act_cols, resp_var):
#            self.net.results['epiA'] = self.net.results['epiA'] * (self.net.results['triatype'] == 'old')
            self.net.smry = self.net.results.groupby(self.group_vars)[act_cols].mean()
            self.subj_obs_data = self.obs_data.loc[self.net.smry.index][resp_var]
            self.net.smry = pd.DataFrame(self.net.smry)
            opt = sp.optimize.fmin(func = self.activation_error, 
                                   x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT,
                                         self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                                   args=(act_cols,
                                         resp_var), 
                                   maxfun=500, full_output=True, disp=self.verbose,
                                   ftol = 0.001, xtol=0.001)
            # save results with fitted parameters
            resp_pred = activation_to_response([opt[0][0],opt[0][1]], self.net.results.epiA, 0) 
            self.net.results['old_rem_pred'] = resp_pred
            # predictions based on semB
            resp_pred = activation_to_response([opt[0][2],opt[0][3]], self.net.results['stim1.semB'], self.net.results['old_rem_pred']) 
            self.net.results['old_know_pred'] = resp_pred
            #combined predictions 
            self.net.results[resp_var+'_pred'] = self.net.results['old_rem_pred']+self.net.results['old_know_pred']             
            return opt   
    
    def activation_error(self, par, act_cols, resp_var):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        pred_data = self.net.smry[act_cols]
        obs = self.subj_obs_data
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs and par[2] > lt and par[3] > ls and par[2] < ht and par[3] < hs
        if within_bounds:
            # predictions based on epiA
            resp_pred = activation_to_response([par[0],par[1]], pred_data.epiA, 0) 
            self.net.smry['old_rem_pred'] = resp_pred
            # predictions based on semB
            resp_pred = activation_to_response([par[2],par[3]], pred_data['stim1.semB'], resp_pred) 
            self.net.smry['old_know_pred'] = resp_pred
            #combined predictions 
            resp_pred = self.net.smry['old_rem_pred']+self.net.smry['old_know_pred']
            return rmse(obs.values, resp_pred)
        return 1           
        
            
class RemKnowModel(Model):
    def __init__(self, nets, init_pars,  shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, fit_filter=None, 
                 reject_related=False, verbose=True):
        self.resp_vars = ['rem','know']
        Model.__init__(self, nets, init_pars, shared_par_names, subject_par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       reject_related, verbose)
        
    def run_net_and_fit_activation_pars(self, pars, act_col):
        self.net.extract_prior_base(self.stim_info)
        self.net.concepts.Bprior = self.net.cPriorB
        self.net.run_trials(pars, self.equations)
        self.filter_results()
        opt_rem = self.fit_activation_to_response('epiA', 'rem')
        opt_know = self.fit_activation_to_response('stim1.semB', 'know', 'rem')
        opt = [[x]+[y] for x,y in zip(opt_know,opt_rem)]
        opt[1] = np.mean(opt[1])
        return({'rem': opt_rem, 'know': opt_know})
        
    def fit_activation_to_response(self, act_col, resp_var, other_col=0):
        if type(other_col) == str:
            if self.reject_related:
                self.net.smry = self.net.results.groupby(self.group_vars+['sameNode'])[act_col, other_col+'_pred'].mean()
                self.subj_obs_data = self.obs_data.loc[self.net.smry.index.droplevel('sameNode')]
            else: 
                self.net.smry = self.net.results.groupby(self.group_vars)[act_col, other_col+'_pred'].mean()
                self.subj_obs_data = self.obs_data.loc[self.net.smry.index]
            other = self.net.results[other_col+'_pred'].values
        else:
            if self.reject_related:
                self.net.smry = self.net.results.groupby(self.group_vars)[act_col].mean()
            self.net.smry = self.net.results.groupby(self.group_vars+['sameNode'])[act_col].mean()
            self.subj_obs_data = self.obs_data.loc[self.net.smry.index.droplevel('sameNode')][resp_var]
            other = 0
            self.net.smry = pd.DataFrame(self.net.smry)
        opt = sp.optimize.fmin(func = self.activation_error, 
                               x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                               args=(act_col,
                                     resp_var,
                                     other_col), 
                               maxfun=500, full_output=True, disp=self.verbose,
                               ftol = 0.001, xtol=0.001)
        self.net.results[resp_var+'_pred'] = activation_to_response(par = opt[0], 
                               x = self.net.results[act_col].values, 
                               other = other)
        if self.reject_related and resp_var == 'rem':
            pred = activation_to_response(par = opt[0], 
                            x = self.net.results[act_col].values, 
                               other = other)
            self.net.results[resp_var+'_pred'] = pred * self.net.results['sameNode']
            
        return opt    

    def activation_error(self, par, act_col, resp_var, other):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        pred_data = self.net.smry[act_col]
        if type(other) == str:
            other = self.net.smry[other+'_pred']
            obs = self.subj_obs_data[resp_var]
        else:
            obs = self.subj_obs_data
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs
        if within_bounds:
            resp_pred = activation_to_response(par, pred_data.values, other) 
            if self.reject_related and resp_var == 'rem':
                self.net.smry = pd.DataFrame(self.net.smry)
                self.net.smry['resp_pred'] = resp_pred * self.net.smry.reset_index(['sameNode'])['sameNode'].values
                resp_pred = self.net.smry.groupby(self.group_vars)['resp_pred'].mean()
            return rmse(obs.values, resp_pred)
        return 1   

class Diana2006RecognitionModel(Model):
    def __init__(self, nets, init_pars, shared_par_names, subject_par_names,
                 par_ranges, equations, group_vars, target_obs, resp_vars, 
                 fit_filter=None, reject_related=False, verbose=True):
        self.resp_vars = resp_vars
        Model.__init__(self, nets, init_pars, shared_par_names, subject_par_names,
                       par_ranges, equations, group_vars, target_obs, fit_filter,
                       reject_related, verbose)
        
    def run_net_and_fit_activation_pars(self, pars, act_col):
        self.net.run_trials(pars, self.equations)
        self.filter_results()
        opt = self.fit_activation_to_response(['epiA'], self.resp_vars[0]) 
        return({'rem': [opt[0][0:2],opt[1]], 'know': [opt[0][2:4],opt[1]]})
        
    def fit_activation_to_response(self, act_cols, resp_var):
            self.net.results['epiA'] = self.net.results['epiA'] * (self.net.results['triatype'] == 'old')
            self.net.smry = self.net.results.groupby(self.group_vars)[act_cols].mean()
            self.subj_obs_data = self.obs_data.loc[self.net.smry.index][resp_var]
            self.net.smry = pd.DataFrame(self.net.smry)
            opt = sp.optimize.fmin(func = self.activation_error, 
                                   x0 = [self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT,
                                         self.ACT_THRESHOLD_INIT, self.ACT_SD_INIT], 
                                   args=(act_cols,
                                         resp_var), 
                                   maxfun=500, full_output=True, disp=self.verbose,
                                   ftol = 0.001, xtol=0.001)
            # save results with fitted parameters
            resp_pred = activation_to_response([opt[0][0],opt[0][1]], self.net.results.epiA, 0) 
            self.net.results['old_rem_pred'] = resp_pred
            # predictions based on semB
            resp_pred = activation_to_response([opt[0][2],opt[0][3]], self.net.results['stim1.semB'], self.net.results['old_rem_pred']) 
            self.net.results['old_know_pred'] = resp_pred
            #combined predictions 
            self.net.results[resp_var+'_pred'] = self.net.results['old_rem_pred']+self.net.results['old_know_pred']             
            return opt   
    
    def activation_error(self, par, act_cols, resp_var):
        """ estimates the RMSE between observations and predictions, 
        averaged over groupVars. If par is outside bounds, return error 1
        """
        pred_data = self.net.smry[act_cols]
        obs = self.subj_obs_data
        lt, ht = self.ACT_THRESHOLD_BOUNDS
        ls, hs = self.ACT_SD_BOUNDS
        within_bounds = par[0] > lt and par[1] > ls and par[0] < ht and par[1] < hs and par[2] > lt and par[3] > ls and par[2] < ht and par[3] < hs
        if within_bounds:
            # predictions based on epiA
            resp_pred = activation_to_response([par[0],par[1]], pred_data.epiA, 0) 
            self.net.smry['old_rem_pred'] = resp_pred
            # predictions based on semB
            resp_pred = activation_to_response([par[2],par[3]], pred_data['stim1.semB'], resp_pred) 
            self.net.smry['old_know_pred'] = resp_pred
            #combined predictions 
            resp_pred = self.net.smry['old_rem_pred']+self.net.smry['old_know_pred']
            return rmse(obs.values, resp_pred)
        return 1         
        
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import scipy as sp
    import sac.main
    import sac.utils
    from importlib import reload
    import matplotlib.pyplot as plt
    from plotnine import *
    import os
    reload(sac.equations)
    reload(sac.main)
    reload(sac.utils)
    
    os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')
    
    def plot_predictions(results, allsubjects, group_vars, pred_var):
        obs = (allsubjects.
            query(filter_cond).
            groupby(group_vars)
            ['CuedRecall'].
            mean().
            reset_index().
            melt(id_vars=group_vars, value_vars=['CuedRecall']))
        pred = (results.
            query(filter_cond).
            groupby(group_vars)
            [pred_var].
            mean().
            reset_index().
            melt(id_vars=group_vars, value_vars=[pred_var]))
        results = pd.concat([pred, obs], axis=0)
        return(ggplot(aes('MemCuePostconsec.value','value', color='variable'),
                      data=results) +
             stat_summary(geom='point') +
             stat_summary(geom='line') +
             facet_wrap('~CueType'))
    
    
    """ DEFINE CONSTANTS TO USE IN THE MODEL """
    initPars = {'SCALE': 1000.0,
       'W': 3.7,
       'contextAct': 1,
       'decaytype': 'power',
       'dl': -0.12,
       'dn': -0.18,
       'fan': 2,
       'p': 0.65,
       'p_forget_prop': 0.1,
       'w_act': 1,
       'prior0': 0.4,
       'prior1': 0.2,
       'prior2': -0.1,
       'sem_theta': 1,
       'w_recovery_rate': 1.02,
       'y': 0.2}
    
    """ SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
    equations = {'conceptLearning': 'learning_equation4',
                  'episodeLearning': 'learning_equation9',
                  'linkLearning': 'learning_equation2',
                  'conceptDecay': 'decay_power',
                  'episodeDecay': 'decay_power',
                  'linkDecay': 'decay_power'}
    
    filter_cond = ('procedure == "test" ' +
                  'and MemCuePostprioritem1 != "nan" ' +
                  'and MemCuePostconsec < 4')
    
    """ LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
    allsubjects = sac.utils.load_trials('data/marevic2017_exp1.csv', scale=initPars['SCALE'])
    trials = sac.utils.select_subjects(allsubjects, range(3))
    stim_columns = ['stim1','stim2','list']  
    stimInfo = sac.utils.get_stim_info(allsubjects, stim_columns, initPars)
    nets = sac.utils.init_networks(trials, ['subject'], stim_columns, initPars, stimInfo, duration=True)
    group_vars = ['CueType','MemCuePostconsec.value']

    model = CuedRecallModel(nets=nets, 
                            init_pars=initPars, 
                            shared_par_names=['w_recovery_rate','W'],
                            subject_par_names=None,
                            par_ranges={'w_recovery_rate': slice(0.9,1.1,0.1),
                                                      'W': slice(3,5,1)},
                            equations=equations, 
                            group_vars=group_vars, 
                            target_obs=allsubjects, 
                            resp_vars=['CuedRecall'], 
                            fit_filter=filter_cond,
                            verbose=False)
    model.estimate_parameters(order='joint', func_type='brute', act_par_func=None, duration=True)
    results = sac.utils.extract_results(nets)
    
#    op = model.opt_pars
    plot_predictions(results, allsubjects, ['CueType','MemCuePostconsec.value'], 'CuedRecall_pred')
#    
    
    obs = (allsubjects.
        query(filter_cond).
        groupby(group_vars)
        ['CuedRecall'].
        mean())
    pred = (results.
        query(filter_cond).
        query('subject==9').
        groupby(group_vars)
        ['CuedRecall_pred'].
        mean())
    rmse(obs,pred)
#    import warnings
#    warnings.filterwarnings("ignore")
#    import scipy.stats
#    import numpy as np
#    from matplotlib import pyplot as plt
#    import importlib
#    from plotnine import * 
#    import sac.main
#    import sac.utils
#    importlib.reload(sac.equations)
#    importlib.reload(sac.main)
#    importlib.reload(sac.utils)
#    
#    from pdb import set_trace as bp
#    import os
#    
#    os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')
#    
#    """ DEFINE CONSTANTS TO USE IN THE MODEL """
#    initPars = {'SCALE': 1000.0,  # time scale to transform into seconds
#                 'dn': -0.2,           # node pwer decay parameter
#                 'dl':- 0.12,           # link power decay parameter
#                 'p': 0.8,            # learning paramter
#                 'y': 0.2,            # current activation decay parameter
#                 'sem_theta': 1,
#                 'contextAct': 1,
#                 'prior0': 0.4,
#                 'prior1': 0.2,
#                 'prior2': -0.1,
#                 'fan': 2,
#                 'decaytype': 'power',
#                 'W': 5,
#                 'w_recovery_rate': 2,
#                 'w_act': 1
#                 }
#    
#    """ SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
#    equations = {'conceptLearning': 'learning_equation4',
#                  'episodeLearning': 'learning_equation4',
#                  'linkLearning': 'learning_equation2',
#                  'conceptDecay': 'decay_power',
#                  'episodeDecay': 'decay_power',
#                  'linkDecay': 'decay_power'}
#    
#    """ LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
#    allsubjects = sac.utils.load_trials('data/reder2000_exp1.csv', scale=initPars['SCALE'])
#    allsubjects['rem'] = (allsubjects.resp == 'rem').astype(int)
#    allsubjects['know'] = (allsubjects.resp == 'know').astype(int)
#    trials = sac.utils.select_subjects(allsubjects, [13])
#    stim_columns = ['stim','list']
#    
#    stimInfo = sac.utils.get_stim_info(allsubjects, stim_columns, initPars)
#    nets = sac.utils.init_networks(trials, ['subject'], stim_columns, initPars, stimInfo, duration=True)
#    group_vars = ['rep','freq']
#    obs_for_pred = allsubjects.groupby(['subject']+group_vars)[['rem','know']].mean() 
#    model = RemKnowModel(nets, initPars, equations, group_vars, obs_for_pred)
#    model.test(duration=True)
    
    
    
                
    