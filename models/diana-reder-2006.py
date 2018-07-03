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
               'W': 3,
               'contextAct': 1,
               'decaytype': 'power',
               'dl': -0.12,
               'dn': -0.18,
               'fan': 0.2,
               'p': 0.8,
               'prior0': 0.4,
               'prior1': 0.2,
               'prior2': -0.1,
               'sem_theta': 1,
               'w_recovery_rate': 1.14,
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
allsubjects = sac.utils.load_trials('data/formatted_for_modeling/diana2006_exp1_fake.csv', scale=init_pars['SCALE'])
allsubjects = allsubjects.rename(columns={'fam': 'know', 'type':'triatype'})
allsubjects['old'] = allsubjects['rem']+allsubjects['know']
trials = sac.utils.select_subjects(allsubjects, None)
allsubjects = allsubjects.query('source != "fake"')

stim_columns = ['list', 'stim1','stim2']
stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
stim_info = (stim_info.join(allsubjects[allsubjects['procedure'] == 'study'][['stim2','freq']].
                            drop_duplicates().
                            set_index('stim2'), 
                            lsuffix='_'))
stim_info['priorBase'][stim_info['freq'] == 'low'] = 0.2
stim_info['priorBase'][stim_info['freq'] == 'high'] = 0.4
stim_info['priorBase'][stim_info['freq'].isnull()] = 0.3
stim_info['priorFan'][stim_info['freq'] == 'low'] = 0.4
stim_info['priorFan'][stim_info['freq'] == 'high'] = 0.8
stim_info['priorFan'][stim_info['freq'].isnull()] = 0.6

split_nets_by = ['subject']
group_vars = ['triatype','freq','freq_prioritem1']
filter_cond = '(procedure=="test") and (~ ((freq_prioritem1 == "nan") and (triatype == "old")))'
filter_cond = '(procedure=="test") and (freq_prioritem1 != "nan")'

# initialize sac networks
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

# run sac network for each subject
for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)

fit = results.query(filter_cond).groupby(group_vars)['epiA'].mean().reset_index()

#%% 

def plot_func(w):
    init_pars['w_recovery_rate'] = w
#    nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
#                               init_pars, stim_info, duration=True)
    for net in nets.values():
        net.run_trials(init_pars, equations)
    results = sac.utils.extract_results(nets)
    fit = results.query(filter_cond).groupby(group_vars)['epiA'].mean().reset_index()
    res = (results.melt(id_vars=['epiA','procedure','freq'], 
                        value_vars=['freq_prioritem1',
                                    'freq_prioritem2',
                                    'freq_prioritem3',
                                    'freq_prioritem4']))
    f1 = (ggplot(aes(x='variable',y='epiA', color='value', group='value'), 
            data=res.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"')) +
         stat_summary(geom='point') +
         stat_summary(geom='line')+
         theme_classic())
    
    f2 = (ggplot(aes(x='freq', y='epiA', color='freq_prioritem1', group='freq_prioritem1'), 
            data=results.query(filter_cond+'and epiB > 0')) +
         stat_summary(geom='point') +
         stat_summary(geom='line') +
         theme_classic() +
         xlab('Frequency of words in the word pair'))  
    
    f3 = (ggplot(aes(x='freq_consec_value', y='epiA', color='freq'), 
        data=results.query(filter_cond)) +
     stat_summary(geom='point') +
     stat_summary(geom='line') +
     theme_classic())
    f2.draw()
    f1.draw()
    f3.draw()
    return(1)
    

#%%
(ggplot(aes(x='freq', y='epiA', color='freq_prioritem1', group='freq_prioritem1'), 
        data=results.query(filter_cond+'and epiB > 0')) +
     stat_summary(geom='point') +
     stat_summary(geom='line') +
     theme_classic() +
     xlab('Frequency of words in the word pair'))

    
#%%

(ggplot(aes(x='freq_consec_value', y='epiA'), 
        data=results.query(filter_cond+'and freq=="low"')) +
     stat_summary(geom='point') +
     stat_summary(geom='line') +
     theme_classic())

#%%
res = (results.melt(id_vars=['epiA','procedure','freq'], 
                    value_vars=['freq_prioritem1',
                                'freq_prioritem2',
                                'freq_prioritem3',
                                'freq_prioritem4']))
(ggplot(aes(x='variable',y='epiA', color='value', group='value'), 
        data=res.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"')) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())
    
#%%
(ggplot(aes('epiA', fill='freq+freq_prioritem1'), 
        data=results.query(filter_cond+'and epiB > 0')) + 
    geom_density(alpha=0.2)).draw() 


#%%
#nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
#                               init_pars, stim_info, duration=True)

model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(1.09,1.16,0.005),
                                                  'W': slice(3,6,0.2),
                        },              
                        equations=equations, 
                        group_vars=['freq','freq_consec_value'], 
                        target_obs=allsubjects,
                        fit_filter=filter_cond,
                        resp_vars=['old'],
                        verbose=True,
                        reject_related=False,
                        summarise='after')

model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
model.fit_with_final_parameters()
results = model.results

results.to_csv('output/final/diana2006_results_fit_mult.csv')
model.save_pars('diana2006')

#%%


#%%

fit = results.query(filter_cond).groupby(group_vars)['old_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['old'] = fit['old_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars)['old'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])

f1 = (ggplot(aes(x='freq', y='old', color='freq_prioritem1', shape='Data', 
                 linetype='Data', group='freq_prioritem1+Data'), data=res.query('triatype=="old"')) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic() +
     xlab('Frequency of words in the word pair') +
     ylab('P(old)')) 
f1.draw()

#%%
group_vars1 = ['freq_consec_value','freq','triatype']
fit = results.query(filter_cond).groupby(group_vars1)['old_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['old'] = fit['old_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars1)['old'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])

f1 = (ggplot(aes(x='freq_consec_value', y='old', shape='Data', 
                 linetype='Data'), data=res.query('triatype=="old"')) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic() +
     xlab('Frequency of words in the word pair') +
     ylab('P(old)')) 
f1.draw()

#%%
fit = (results.melt(id_vars=['old_pred','procedure','freq'], 
                    value_vars=['freq_prioritem1',
                                'freq_prioritem2',
                                'freq_prioritem3',
                                'freq_prioritem4']))
fit = fit.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"').groupby(['variable','value'])['old_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['old'] = fit['old_pred']  

res = (allsubjects.melt(id_vars=['old','procedure','freq'], 
                    value_vars=['freq_prioritem1',
                                'freq_prioritem2',
                                'freq_prioritem3',
                                'freq_prioritem4']))
res = res.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"').groupby(['variable','value'])['old'].mean().reset_index()
res['Data'] = 'observed'
res = pd.concat([fit,res])
    
(ggplot(aes(x='variable',y='old', color='value', shape='Data', linetype='Data', group='Data+value'), 
        data=res) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())


#%%

fit = results.query(filter_cond).groupby(group_vars)['old_rem_pred','old_know_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['rem'] = fit['old_rem_pred']
fit['know'] = fit['old_know_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars)['rem','know'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])
res['old']=res['know']+res['rem']

f1 = (ggplot(aes(x='freq', y='old', color='freq_prioritem1', shape='Data', 
                 linetype='Data', group='freq_prioritem1+Data'), data=res.query('triatype=="old"')) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic() +
     xlab('Frequency of words in the word pair') +
     ylab('P(old)')) 
f1.draw()

#%%
group_vars1 = ['freq_consec_value','freq','triatype']
fit = results.query(filter_cond+'and freq == "low"').groupby(group_vars1)['old_rem_pred','old_know_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['rem'] = fit['old_rem_pred']
fit['know'] = fit['old_know_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars1)['rem','know'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])
res['old']=res['know']+res['rem']

f1 = (ggplot(aes(x='freq_consec_value', y='old', shape='Data', 
                 linetype='Data'), data=res.query('triatype=="old"')) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic() +
     xlab('Frequency of words in the word pair') +
     ylab('P(old)')) 
f1.draw()

#%%
fit = (results.melt(id_vars=['old_rem_pred','old_know_pred','procedure','freq'], 
                    value_vars=['freq_prioritem1',
                                'freq_prioritem2',
                                'freq_prioritem3',
                                'freq_prioritem4']))
fit = fit.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"').groupby(['variable','value'])['old_rem_pred','old_know_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['rem'] = fit['old_rem_pred']
fit['know'] = fit['old_know_pred']   

res = (results.melt(id_vars=['rem','know','procedure','freq'], 
                    value_vars=['freq_prioritem1',
                                'freq_prioritem2',
                                'freq_prioritem3',
                                'freq_prioritem4']))
res = res.query('procedure == "test" & (value == "low" | value == "high") & freq == "low"').groupby(['variable','value'])['rem','know'].mean().reset_index()
res['Data'] = 'observed'
res = pd.concat([fit,res])
res['old']=res['know']+res['rem']
    
(ggplot(aes(x='variable',y='old', color='value', shape='Data', linetype='Data', group='Data+value'), 
        data=res) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())

