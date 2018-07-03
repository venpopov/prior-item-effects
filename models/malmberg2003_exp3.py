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
def gen_trial_list(hf_num, lf_num, listn):
    """ generates one list with desired number of hf and lf words """
    hfwords = sac.utils.generate_words(100, 100, 1000)
    lfwords = sac.utils.generate_words(100, 0, 5)
    hf1 = hfwords.iloc[0:hf_num]
    hf2 = hfwords.iloc[hf_num:(2*hf_num)]
    hf3 = hfwords.iloc[(2*hf_num):(3*hf_num)]
    hf4 = hfwords.iloc[(3*hf_num):(4*hf_num)]
    lf1 = lfwords.iloc[0:lf_num]
    lf2 = lfwords.iloc[lf_num:(2*lf_num)]
    lf3 = lfwords.iloc[(2*lf_num):(3*lf_num)]
    lf4 = lfwords.iloc[(3*lf_num):(4*lf_num)]
    hfhf = hf1
    hfhf['stim2'] = hf2['word'].tolist()
    hflf = hf3
    hflf['stim2'] = lf1['word'].tolist()
    lfhf = lf2
    lfhf['stim2'] = hf4['word'].tolist()
    lflf = lf3
    lflf['stim2'] = lf4['word'].tolist()
    hfhf['target_freq'] = 'hf'
    hfhf['partner_freq'] = 'hf'
    hflf['target_freq'] = 'hf'
    hflf['partner_freq'] = 'lf'
    lfhf['target_freq'] = 'lf'
    lfhf['partner_freq'] = 'hf'
    lflf['target_freq'] = 'lf'
    lflf['partner_freq'] = 'lf'
    
    hfhf['duration'] = 1.450
    lflf['duration'] = 1.450
    hflf['duration'] = 1.450
    lfhf['duration'] = 1.450
    hfhf['duration'][0:int(hf_num/2)] = 4.250
    hflf['duration'][0:int(hf_num/2)] = 4.250
    lfhf['duration'][0:int(hf_num/2)] = 4.250
    lflf['duration'][0:int(hf_num/2)] = 4.250
    
    hflf = hflf.sample(frac=0.5)
    lfhf = lfhf.sample(frac=0.5)
#    study_idx = np.random.choice(np.arange(hf_num), replace=False, size=int(hf_num/2))
    study = pd.concat([hfhf,hflf,lfhf,lflf])
    study['procedure'] = 'study'
    study = study.sample(frac=1)
    study['sp'] = np.arange(study.shape[0])+1
    test = study.copy()
    test['procedure'] = 'test'
    test = test.sample(frac=1)
    test['study_duration'] = test['duration']
    test['duration'] = 0
    test['duration'].iloc[0] = 31
    test['stim2'] = 'nan'
#    test['type'] = 'new'
#    test['type'][test['word'].isin(study['word'])] = 'old'
    dat = pd.concat([study, test])
    dat['trial'] = np.arange(dat.shape[0]) + 1
    dat['list'] = 'context' + str(listn)
    dat = dat.rename(columns = {'word': 'stim1'})
    return(dat)
    
def gen_trials(n_subjects, nlists):
    """ generates trials to be used for modelling for n_subjects*10 """
    trials = pd.DataFrame()
    for subject in np.arange(n_subjects):
        for listn in range(nlists):
            dat1 = gen_trial_list(16,16, 1)
            dat1['subject'] = subject
            trials = pd.concat([trials, dat1])
        trials['trial'] = np.arange(trials.shape[0]) +1
        trials['onset'] = trials['duration'].cumsum()
    return(trials)
    
#%%
#### The trials were originally generated with this command, but it takes
#### several minutes, so just load the presaved dataframe
np.random.seed(1274589)
trials = gen_trials(100, 1)
#trials.to_csv('data/mixed_proportion_simultated_trials.csv')
#trials = pd.read_csv('data/mixed_proportion_simultated_trials.csv')

#data = pd.DataFrame(dict(freq = ['lf','lf','lf','hf','hf','hf'],
#                         type = ['0% HF lists','25% HF lists','75% HF lists',
#                                 '25% HF lists','75% HF lists', '100% HF lists'],
#                         acc = [0.48, 0.51, 0.52, 0.42, 0.48, 0.62],
#                         procedure = 'test'))

""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1.0,
             'W': 5,
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
             'w_recovery_rate': 0.91,
             'y': 0.8,
             'multiply_activation': True}


""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation10',
             'episodeLearning': 'learning_equation10',
             'linkLearning': 'learning_equation2',
             'conceptDecay': 'decay_power',
             'episodeDecay': 'decay_power',
             'linkDecay': 'decay_power'}
    
filter_cond = ('procedure == "test"')
stim_columns = ['stim1','stim2','list'] 
group_vars = ['partner_freq','target_freq','study_duration'] 
split_nets_by = ['subject','list']

trials['onset'] = trials['onset']/init_pars['SCALE']
#trials = sac.utils.select_subjects(trials, np.arange(10)+1)
stim_info = sac.utils.get_stim_info(trials, stim_columns, init_pars)
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

#%%
init_pars['w_recovery_rate'] = 1
#init_pars['y'] = 0.6
for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.527, 0.356], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')

fit = results.query(filter_cond).groupby(group_vars)['epiA','epiB'].mean().reset_index()
(ggplot(aes('partner_freq','epiA', color='target_freq', group='target_freq'), data=fit) +
     geom_point() +
     geom_line() +
     facet_wrap('study_duration'))

#%%
data = pd.DataFrame(dict(target_freq = ['hf','hf','hf','hf','lf','lf','lf','lf'],
                         partner_freq = ['hf','hf','lf','lf','hf','hf','lf','lf'],
                         study_duration = [1.450,4.250]*4,
                         hits = [0.51, 0.59, 0.44, 0.58,0.68,0.67,0.55,0.69],
                         procedure = 'test'))

(ggplot(aes('partner_freq','hits', color='target_freq', group='target_freq'), data=data) +
     geom_point() +
     geom_line() +
     facet_wrap('study_duration'))

#%%
init_pars['w_recovery_rate'] = 0.95
#init_pars['W'] = 4.5
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(0.95,0.96,0.005),
                        },              
                        equations=equations, 
                        group_vars=['target_freq','partner_freq','study_duration'], 
                        target_obs=data,
                        fit_filter='procedure == "test"',
                        resp_vars=['hits'],
                        verbose=True,
                        reject_related=False,
                        summarise='after')

model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
#model.fit_with_final_parameters()
results = model.results

#%%
pred = model.pred_smry
pred = pd.merge(pred, model.obs_data.reset_index(), how='left')
pred = pred.melt(id_vars = ['target_freq','partner_freq','study_duration'], value_vars=['hits_pred','hits'])

(ggplot(aes('partner_freq','value', color='target_freq', group='target_freq+variable', linetype='variable', shape='variable'), data=pred) +
     geom_point() +
     geom_line() +
     facet_wrap('study_duration'))

#%%
act_pars = [model.opt_pars['epiA_act_theta'], model.opt_pars['epiA_act_sd']]
model.results['hits_pred'] = sac.fit_model_group.activation_to_response(act_pars,
                  model.results['epiA'])

results1 = pd.merge(model.results, data, how='left')
results1.to_csv('output/final/malmberg2002_exp3_sim1.csv')
model.save_pars('malmberg2002_exp31')

