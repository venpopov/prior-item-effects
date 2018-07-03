# # <codecell>
import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import random

import warnings 
warnings.simplefilter("ignore")
from plotnine import *

from pdb import set_trace as bp

import sac.main
import sac.utils
import sac.fit_model_group

from importlib import reload
reload(sac.equations)
reload(sac.main)
reload(sac.utils)
reload(sac.fit_model_group)

os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')


#%% genereate trials
def gen_trial_list(hf_num, lf_num, r_int, listn, alternating='no'):
    """ generates one list with desired number of hf and lf words """
    hfwords = sac.utils.generate_words(hf_num, 100, 1000)
    hfwords['freq'] = 'W'
    hfwords['stim3'] = ['hfword_' + str(i) for i in np.arange(hfwords.shape[0])]
    lfwords = sac.utils.generate_words(lf_num, 0, 3)
    lfwords['freq'] = 'NW'
    lfwords['stim3'] = ['lfword_' + str(i) for i in np.arange(lfwords.shape[0])]
    study = pd.concat([hfwords,lfwords])
    study['procedure'] = 'study'
    study = study.sample(frac=1)
    
    if alternating != 'no':
        study = study.sort_values(by = 'freq')
        nrow = study.shape[0]
        if alternating == 'W-NW':
            idx1 = [int(nrow/2),0]
        elif alternating == 'NW-W':
            idx1 = [0,int(nrow/2)]
        idx = []
        for i in range(int(nrow/2)):
            idx2 = [x+i for x in idx1]
            idx = idx+idx2
        study = study.iloc[idx]      
    
    study['stim2'] = ['position_' + str(x+1) for x in np.arange(study.shape[0])]
    study['sp'] = np.arange(study.shape[0])+1
    
    test = study.copy()
    test['procedure'] = 'test'
#    test = test.sample(frac=1)
    test['word'] = 'nan'
    test['stim3'] = 'nan'
    dat = pd.concat([study, test])
    dat['trial'] = np.arange(dat.shape[0]) + 1
    dat['duration'] = 1000
    dat['duration'][(dat['procedure'] == 'study') & (dat['sp'] == 1)] = 5000
    dat['duration'][dat['procedure'] == 'test'] = r_int
    dat['duration'][(dat['procedure'] == 'test') & (dat['sp'] == 1)] = 2000
#    dat['onset'] = (dat['trial']-1) * 1000
#    dat['onset'][dat['procedure'] == 'test'] = max(dat['onset'][dat['procedure']== 'study']) + dat['trial'][dat['procedure'] == 'test']*r_int + 2000
    dat['list'] = 'context' + str(listn)
    dat = dat.rename(columns = {'word': 'stim1'})
    return(dat)
    
def gen_trials(n_subjects, nlists, r_int):
    """ generates trials to be used for modelling for n_subjects*10 """
    trials = pd.DataFrame()
    for subject in np.arange(n_subjects):
        for listn in range(nlists):
            dat1 = gen_trial_list(6, 0, r_int, 4*listn+1)
            dat2 = gen_trial_list(0, 6, r_int, 4*listn+2)
            dat3 = gen_trial_list(3, 3, r_int, 4*listn+3, alternating='W-NW')
            dat4 = gen_trial_list(3, 3, r_int, 4*listn+4, alternating='NW-W')
            dat1['composition'] = 'Pure W'
            dat2['composition'] = 'Pure NW'
            dat3['composition'] = 'Alternating W-NW'
            dat4['composition'] = 'Alternating NW-W'
            dat1['subject'] = subject
            dat2['subject'] = subject
            dat3['subject'] = subject
            dat4['subject'] = subject
            trials = pd.concat([trials, dat1, dat2, dat3, dat4])
        trials['trial'] = np.arange(trials.shape[0]) +1
        trials['onset'] = trials['duration'].cumsum()
#        trials['onset'][trials['procedure'] == 'test'] = max(trials['onset'][trials['procedure']== 'study']) + trials['trial'][trials['procedure'] == 'test']*r_int + 000
    trials['testtype'] = ''
    trials['testtype'][trials['procedure'] == 'test'] = ''
    

    return(trials)
    
#%%
#### The trials were originally generated with this command, but it takes
#### several minutes, so just load the presaved dataframe
np.random.seed(12445)
trials = gen_trials(4, 1, 1000)
#trials.to_csv('data/mixed_proportion_simultated_trials.csv')
#trials = pd.read_csv('data/mixed_proportion_simultated_trials.csv')

#data = pd.DataFrame(dict(freq = ['lf','lf','lf','hf','hf','hf'],
#                         type = ['0% HF lists','25% HF lists','75% HF lists',
#                                 '25% HF lists','75% HF lists', '100% HF lists'],
#                         acc = [0.48, 0.51, 0.52, 0.42, 0.48, 0.62],
#                         procedure = 'test'))

""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1000.0,
             'W': 5.87,
             'contextAct': 1,
             'decaytype': 'power',
             'dl': -0.12,
             'dn': -0.18,
             'fan': 0.22,
             'p': 0.8,
             'w_act': 2,
             'prior0': 0.4,
             'prior1': 0.2,
             'prior2': -0.1,
             'sem_theta': 1,
             'w_recovery_rate': 0.215,
             'y': 0.8,
             'multiply_activation': True}


""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation12',
             'episodeLearning': 'learning_equation12',
             'linkLearning': 'learning_equation2',
             'conceptDecay': 'decay_power',
             'episodeDecay': 'decay_power',
             'linkDecay': 'decay_power'}
    
filter_cond = ('procedure == "test"')
stim_columns = ['stim1','list','stim2','stim3'] 
group_vars = ['freq','sp','composition'] 
split_nets_by = ['subject','list']

trials['onset'] = trials['onset']/init_pars['SCALE']
#trials = sac.utils.select_subjects(trials, np.arange(10)+1)
stim_info = sac.utils.get_stim_info(trials, stim_columns, init_pars)
#stim_info['priorBase'][stim_info['freq'] <= 3] = 0
#stim_info['priorFan'][stim_info['freq'] <= 3] = 0
stim_info['priorBase'][['hfword' in i for i in stim_info['Word']]] = 0.99
stim_info['priorBase'][['lfword' in i for i in stim_info['Word']]] = 0.2
stim_info['priorFan'][['hfword' in i for i in stim_info['Word']]] = 1000
stim_info['priorFan'][['lfword' in i for i in stim_info['Word']]] = 0
stim_info['priorBase'][['position' in i for i in stim_info['Word']]] = 0.99
stim_info['priorBase'][['position' in i for i in stim_info['Word']]] = 0.8
stim_info['priorBase'][['nan' in i for i in stim_info['Word']]] = 0.99
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

#%%
init_pars['w_recovery_rate'] = 2
#init_pars['y'] = 0.6
for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.527, 0.356], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')

fit = results.query(filter_cond).groupby(group_vars)['epiA','epiB'].mean().reset_index()
(ggplot(aes('sp','epiA', color='composition', group='composition', shape='freq'), data=fit) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
#init_pars['W'] = 3
#init_pars['w_recovery_rate'] = 1
for net in nets.values():
    net.run_trials(init_pars, equations)
#    
results = sac.utils.extract_results(nets)
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.527, 0.356], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')

fit = results.query('procedure == "study"').groupby(['composition','sp'])['epiS'].mean().reset_index()
(ggplot(aes('sp','epiS', color='composition', group='composition'), data=fit) +
     stat_summary() +
     stat_summary(geom='line'))
#%%
#results = sac.utils.extract_results(nets)
results['acc_pred'] = sac.fit_model_group.activation_to_response([0.383, 0.539], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')

fit = results.query(filter_cond).groupby(group_vars)['epiA','epiB','acc_pred'].mean().reset_index()
(ggplot(aes('sp','acc_pred', color='composition', group='composition'), data=fit) +
     stat_summary() +
     stat_summary(geom='line') +
     coord_cartesian(ylim=[0,1]))

#%%
data = pd.DataFrame(dict(sp = [1,2,3,4,5,6]*4,
                         composition = ['Pure W']*6 + 
                                       ['Pure NW']*6 + 
                                       ['Alternating W-NW']*6 + 
                                       ['Alternating NW-W']*6,
                         freq = ['W']*6 + ['NW']*6 + ['W','NW']*3 + ['NW','W']*3,
                         acc = [0.97,0.88,0.82,0.75,0.66,0.85,
                                0.85,0.68,0.55,0.41,0.32,0.48,
                                0.96,0.81,0.81,0.59,0.56,0.65,
                                0.89,0.87,0.69,0.71,0.50,0.80],
                         procedure = 'test',
                         experiment = 2))

    
(ggplot(aes('sp','acc',shape='freq', color='composition', group='composition'), data=data)+
 stat_summary(geom='point') +
 stat_summary(geom='line') +
 coord_cartesian(ylim=[0,1]))

#%%
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate','W'],
                        par_ranges={'w_recovery_rate': slice(0.8,0.81,0.005),
                                                  'W': slice(3,6,0.2),
                        },              
                        equations=equations, 
                        group_vars=['freq','sp','composition'], 
                        target_obs=data,
                        fit_filter='procedure == "test" and sp != 6',
                        resp_vars=['acc'],
                        verbose=True,
                        reject_related=False,
                        summarise='after')

model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
#model.fit_with_final_parameters()
results = model.results

#%%
pred = model.pred_smry
pred = pd.merge(pred, model.obs_data.reset_index(), how='left')
pred = pred.melt(id_vars = ['freq','sp','composition'], value_vars=['acc_pred','acc'])

(ggplot(aes('sp','value',color='composition', shape='freq', group='composition', linetype='variable'), data=pred)+
 geom_point()+
 geom_line()+
# coord_cartesian(ylim=[0,1]) +
 theme_classic() +
 facet_wrap('variable'))

#%%
act_pars = [model.opt_pars['epiA_act_theta'], model.opt_pars['epiA_act_sd']]
model.results_full['acc_pred'] = sac.fit_model_group.activation_to_response(act_pars,
                  model.results_full['epiA'])

results1 = pd.merge(model.results_full, data, how='left')
results1.to_csv('output/final/hulme2003_simulation_exp3.csv')
model.save_pars('hulme2003_exp3')