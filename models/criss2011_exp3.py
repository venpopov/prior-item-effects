import sys
import pandas as pd
import numpy as np
import random
import scipy as sp
import sac.utils
import sac.main
import sac.fit_model_group
from importlib import reload
import matplotlib.pyplot as plt
from plotnine import *
#from pandas_ply import install_ply, X, sym_call
#install_ply(pd)
reload(sac.equations)
reload(sac.main)
reload(sac.utils)
reload(sac.fit_model_group)



#%%
""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1000.0,
   'W': 2.7,
   'contextAct': 1,
   'decaytype': 'power',
   'dl': -0.12,
   'dn': -0.18,
   'fan': 0.2,
   'p': 0.8,
   'w_act': 2,
   'prior0': 0.4,
   'prior1': 0.2,
   'prior2': -0.1,
   'sem_theta': 1,
   'w_recovery_rate': 0.55,
   'y': 0.2,
   'multiply_activation': True}

""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation12',
              'episodeLearning': 'learning_equation12',
              'linkLearning': 'learning_equation2',
              'conceptDecay': 'decay_power',
              'episodeDecay': 'decay_power',
              'linkDecay': 'decay_power'}

""" LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
allsubjects = sac.utils.load_trials('data/formatted_for_modeling/criss2011_exp3.csv', scale=init_pars['SCALE'])
#trials = sac.utils.select_subjects(allsubjects, [3,8,24,6,13,1,10])
trials = sac.utils.select_subjects(allsubjects, None)

stim_columns = ['list', 'stim1','stim2']
stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
stim1 = allsubjects[allsubjects['procedure'] == 'study'][['stim1','cuefreq','cv']]
stim2 = allsubjects[allsubjects['procedure'] == 'study'][['stim2','targetfreq','cv']]
stim2.columns = ['stim','freq','cv']
stim1.columns = ['stim','freq','cv']
stim = pd.concat([stim1,stim2]).drop_duplicates().set_index('stim')
stim_info = (stim_info.join(stim, lsuffix='_'))
#stim_info['priorBase'][stim_info['freq'] == 'lf'] = 0.2
#stim_info['priorBase'][stim_info['freq'] == 'hf'] = 0.4
stim_info['priorBase'][stim_info['freq'].isnull()] = 0.3
stim_info['priorBase'][stim_info['nodeType'] == 'context'] = 0.99
#stim_info['priorFan'][stim_info['freq'] == 'lf'] = 0
#stim_info['priorFan'][stim_info['freq'] == 'hf'] = 0.5
stim_info['priorFan'][stim_info['freq'].isnull() ] = 0

split_nets_by = ['subject','list']
group_vars = ['cuefreq','targetfreq','cv']
filter_cond = 'procedure=="test"'

# initialize sac network for each subject
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

# run and fit models
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate','W'],
                        par_ranges={'w_recovery_rate': slice(0.36,0.6,0.005),
                                                  'W': slice(2.5,3.5,0.2),
                                                  'p': slice(0.4,0.7,0.01),
                                                  'fan': slice(0.2,0.21,0.1),
                        },              
                        equations=equations, 
                        group_vars=group_vars, 
                        target_obs=allsubjects,
                        fit_filter=filter_cond,
                        resp_vars=['acc'],
                        verbose=True,
                        reject_related=False,
                        summarise='after')

model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.0001, duration=True)
#model.fit_with_final_parameters()
results = model.results
full = model.results_full
#results.to_csv('output/criss2011_exp3_results_fit_eq12.csv')

obs = allsubjects.query(filter_cond).groupby(group_vars)['acc'].mean().reset_index()
pred = results.query(filter_cond).groupby(group_vars)['acc_pred'].mean().reset_index()
#pred = model.pred_smry
obs['pred'] = pred['acc_pred']
obs = obs.melt(id_vars = group_vars)
(ggplot(aes(x='cuefreq', y='value', color='targetfreq', group='variable+targetfreq', shape='variable', linetype='variable'), data=obs) +
     geom_point() +
     geom_line() +
     facet_wrap('~cv')) 

    #%%
(ggplot(aes(x='cuefreq', y='epiA', color='targetfreq', 
            group='targetfreq+cv', shape='cv', linetype='cv'), 
        data=results.groupby(group_vars)['epiA'].mean().reset_index()) +
     geom_point() +
     geom_line()) 
    

#%%
(ggplot(aes(x='cuefreq', y='epiS', group='targetfreq', color='targetfreq'), 
        data=model.results_full.query('procedure=="study"')) +
     stat_summary()+
#     geom_violin(aes(group='cuefreq+targetfreq')) +
     stat_summary(geom='line'))

#%%
(ggplot(aes(x='cuefreq', y='wmSpent', group='targetfreq', color='targetfreq'), 
        data=model.results_full.query('procedure=="study"')) +
     stat_summary()+
#     geom_violin(aes(group='cuefreq+targetfreq')) +
     stat_summary(geom='line'))

#%%
obs = allsubjects.query(filter_cond).groupby(['cuefreq','targetfreq'])['acc'].mean()
pred = results.query(filter_cond).groupby(['cuefreq','targetfreq'])['acc_pred'].mean()
smry = pd.concat([obs,pred], axis=1)
smry = smry.reset_index().melt(id_vars = smry.index.names)
(ggplot(aes(x='cuefreq', y='value', color='variable', group='variable'), data=smry) +
     geom_point() +
     geom_line() +
     facet_wrap('~targetfreq')) 


#%%
(ggplot(aes(x='cuefreq', y='epiB', group=1), data=fit) +
     stat_summary(geom="point") +
     stat_summary(geom='line') +
     facet_wrap('~targetfreq')) 


#%%
(ggplot(aes(x='cuefreq', y='acc_pred', group=1), data=fitted.query('procedure == "test"')) +
     stat_summary(geom="point") +
     stat_summary(geom='line') +
     facet_wrap('~targetfreq')) 