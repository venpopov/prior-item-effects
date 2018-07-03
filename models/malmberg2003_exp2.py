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
    hfwords = sac.utils.generate_words(hf_num, 100, 1000)
    lfwords = sac.utils.generate_words(lf_num, 0, 5)
    hfwords['freq'] = 'hf'
    lfwords['freq'] = 'lf'
    hfwords['duration'] = 0.25
    lfwords['duration'] = 0.25
    hfwords['duration'][int(hf_num/3):int(2*hf_num/3)] = 1
    lfwords['duration'][int(lf_num/3):int(2*lf_num/3)] = 1
    hfwords['duration'][int(2*hf_num/3):hf_num] = 3
    lfwords['duration'][int(2*lf_num/3):lf_num] = 3   
#    study_idx = np.random.choice(np.arange(hf_num), replace=False, size=int(hf_num/2))
    study = pd.concat([hfwords,lfwords])
    study['procedure'] = 'study'
    study = study.sample(frac=1)
    study['sp'] = np.arange(study.shape[0])+1
    test = pd.concat([hfwords, lfwords])
    test['procedure'] = 'test'
    test = test.sample(frac=1)
    test['study_duration'] = test['duration']
    test['duration'] = 1
    test['duration'].iloc[0] = 31
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
            dat1 = gen_trial_list(60,60, 1)
            dat1['subject'] = subject
            trials = pd.concat([trials, dat1])
        trials['trial'] = np.arange(trials.shape[0]) +1
        trials['onset'] = trials['duration'].cumsum()
    return(trials)
    
#%%
#### The trials were originally generated with this command, but it takes
#### several minutes, so just load the presaved dataframe
np.random.seed(1274589)
trials = gen_trials(10, 1)
#trials.to_csv('data/mixed_proportion_simultated_trials.csv')
#trials = pd.read_csv('data/mixed_proportion_simultated_trials.csv')

#data = pd.DataFrame(dict(freq = ['lf','lf','lf','hf','hf','hf'],
#                         type = ['0% HF lists','25% HF lists','75% HF lists',
#                                 '25% HF lists','75% HF lists', '100% HF lists'],
#                         acc = [0.48, 0.51, 0.52, 0.42, 0.48, 0.62],
#                         procedure = 'test'))

""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1.0,
             'W': 3,
             'contextAct': 1,
             'decaytype': 'power',
             'dl': -0.12,
             'dn': -0.18,
             'fan': 0.4,
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
equations = {'conceptLearning': 'learning_equation12',
             'episodeLearning': 'learning_equation12',
             'linkLearning': 'learning_equation2',
             'conceptDecay': 'decay_power',
             'episodeDecay': 'decay_power',
             'linkDecay': 'decay_power'}
    
filter_cond = ('procedure == "test"')
stim_columns = ['stim1','list'] 
group_vars = ['freq','study_duration'] 
split_nets_by = ['subject','list']

trials['onset'] = trials['onset']/init_pars['SCALE']
#trials = sac.utils.select_subjects(trials, np.arange(10)+1)
stim_info = sac.utils.get_stim_info(trials, stim_columns, init_pars)
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

#%%
#init_pars['w_recovery_rate'] = 0.2
#init_pars['y'] = 0.6
for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.527, 0.356], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')

#%%

fit = results.query(filter_cond).groupby(group_vars)['epiA','epiB'].mean().reset_index()
(ggplot(aes('study_duration','epiA', color='freq', group='freq'), data=fit) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
data = pd.DataFrame(dict(freq = ['hf','hf','hf','lf','lf','lf'],
                          study_duration = [0.25,1,3,0.25,1,3],
                          hits = [0.52,0.52,0.58, 0.55, 0.65,0.74],
                          procedure = 'test'))

(ggplot(aes('study_duration','hits', color='freq', group='freq', shape='type'), data=data) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
#init_pars['w_recovery_rate'] = 0.8
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(0.87,0.88,0.005),
                                                  'W': slice(2,3,0.2),
                                                  'fan': slice(0.2,0.3,0.01)
                        },              
                        equations=equations, 
                        group_vars=['freq','study_duration'], 
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
pred = pred.melt(id_vars = ['freq','study_duration'], value_vars=['hits_pred','hits'])

(ggplot(aes('study_duration','value',color='freq', shape='variable', group='freq+variable', linetype='variable'), data=pred)+
 geom_point()+
 geom_line()+
# coord_cartesian(ylim=[0,1]) +
 theme_classic())

#%%
act_pars = [model.opt_pars['epiA_act_theta'], model.opt_pars['epiA_act_sd']]
model.results_full['hits_pred'] = sac.fit_model_group.activation_to_response(act_pars,
                  model.results_full['epiA'])

results1 = pd.merge(model.results_full, data, how='left')
results1.to_csv('output/final/malmberg2002_exp2_sim.csv')
model.save_pars('malmberg2002')

