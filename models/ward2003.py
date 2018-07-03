import sys
import pandas as pd
import numpy as np
import random
import scipy as sp
import sac.utils
import sac.main
import sac.fit_model_group
import importlib
import matplotlib.pyplot as plt
from plotnine import *
#from pandas_ply import install_ply, X, sym_call
#install_ply(pd)
importlib.reload(sac.equations)
importlib.reload(sac.main)
importlib.reload(sac.utils)
importlib.reload(sac.fit_model_group)



#%%
""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1000.0,
               'W': 2.8,
               'contextAct': 1,
               'decaytype': 'power',
               'dl': -0.12,
               'dn': -0.18,
               'fan': 0.8,
               'p': 0.8,
               'prior0': 0.4,
               'prior1': 0.2,
               'prior2': -0.1,
               'sem_theta': 1,
               'w_recovery_rate': 0.45,
               'y': 0.2,
               'w_act': 2,
               'multiply_activation': True}

""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation10',
              'episodeLearning': 'learning_equation10',
              'linkLearning': 'learning_equation2',
              'conceptDecay': 'decay_power',
              'episodeDecay': 'decay_power',
              'linkDecay': 'decay_power'}


""" LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
allsubjects = sac.utils.load_trials('data/ward2003_exp3.csv', scale=init_pars['SCALE'])
trials = sac.utils.select_subjects(allsubjects, None)
#allsubjects = allsubjects.query('source != "fake"')

stim_columns = ['list', 'stim1']
stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
stim_info = (stim_info.join(allsubjects[allsubjects['procedure'] == 'study'][['stim1','freq']].
                            drop_duplicates().
                            set_index('stim1'), 
                            lsuffix='_'))
#stim_info['priorBase'][stim_info['freq'] == 'low'] = 0.2
#stim_info['priorBase'][stim_info['freq'] == 'high'] = 0.4
#stim_info['priorBase'][stim_info['freq'].isnull()] = 0.3
#stim_info['priorFan'][stim_info['freq'] == 'low'] = 0.4
#stim_info['priorFan'][stim_info['freq'] == 'high'] = 0.8
#stim_info['priorFan'][stim_info['freq'].isnull()] = 0.6

split_nets_by = ['subject','list']
group_vars = ['composition','freq','freq_prioritem1']
filter_cond = '(procedure=="test") and (freq_prioritem1 != "nan")'


nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)

fit = results.query(filter_cond).groupby(group_vars)['epiA','acc'].mean().reset_index()
results.to_csv('output/ward2003_results_for_ploting_wm.csv')

#%%

def plot_func(w):
    init_pars['w_recovery_rate'] = w
    for net in nets.values():
        net.run_trials(init_pars, equations)
    results = sac.utils.extract_results(nets)
    fit = results.query(filter_cond).groupby(group_vars)['epiA','acc'].mean().reset_index()
    f1 = (ggplot(aes(x='freq', y='epiA', color='freq_prioritem1', group='freq_prioritem1'), 
            data=fit) +
         geom_point() +
         geom_line() +
         theme_classic() + 
         xlab('Frequency of words in the word pair') +
         facet_wrap('composition'))
    return(f1)
    
#%%
(ggplot(aes(x='freq', y='acc', color='freq_prioritem1', group='freq_prioritem1'), 
        data=fit) +
     geom_point() +
     geom_line() +
     theme_classic() + 
     xlab('Frequency of words in the word pair') +
     facet_wrap('composition'))
    
#%%
    fit = results.query(filter_cond).groupby(['freq','composition','nominal.sp'])['epiB','acc'].mean().reset_index()
(ggplot(aes(x='nominal.sp', y='epiB', color='freq'), 
        data=fit) +
     geom_point() +
     geom_line() +
     facet_wrap('composition') +
     theme_classic())
    
#%%
#import sys
#sys.stdout = open('output/ward2003_console_fit.txt', 'w')
#import sac.fit_model_group
importlib.reload(sac.fit_model_group)
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(0.45,0.46,0.01),
                                                  'W': slice(1.4,5,0.1),
                        },              
                        equations=equations, 
                        group_vars=['composition','freq','freq_prioritem1'], 
                        target_obs=allsubjects,
                        fit_filter=filter_cond,
                        resp_vars=['acc'],
                        verbose=True,
                        reject_related=False,
                        summarise='after')

#model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
model.fit_with_final_parameters()
results = model.results
#results.to_csv('output/ward2003_results_fit.csv')
#
#nets1 = nets.copy()
#model2 = sac.fit_model_group.Model(nets=nets1, 
#                        init_pars=init_pars, 
#                        par_names=['w_recovery_rate','W'],
#                        par_ranges={'w_recovery_rate': slice(0.3,0.6,0.01),
#                                                  'W': slice(1.4,5,0.1),
#                        },              
#                        equations=equations, 
#                        group_vars=['composition','freq','freq_prioritem1'], 
#                        target_obs=allsubjects,
#                        fit_filter=filter_cond,
#                        resp_vars=['acc'],
#                        verbose=True,
#                        reject_related=False)
#
#model2.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
#results2 = model2.results
#results2.to_csv('output/ward2003_results_fit_fmin.csv')

#%%
fit = results.query(filter_cond).groupby(group_vars)['acc_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['acc'] = fit['acc_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars)['acc'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])

f1 = (ggplot(aes(x='freq', y='acc', color='freq_prioritem1', shape='Data', 
                 linetype='Data', group='freq_prioritem1+Data'), data=res) +
     geom_point() +
     geom_line() +
     theme_classic() +
     facet_wrap('composition'))
f1.draw()


