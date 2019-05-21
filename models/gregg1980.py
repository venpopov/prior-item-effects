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
def gen_list(n, wfcat, wfmin, wfmax, duration, isi, retention_delay, listn):
    words = sac.utils.generate_words(n, wfmin, wfmax)
    words['freq'] = wfcat
    words['duration'] = duration+isi
    study = words.copy()
    study['procedure'] = 'study'
    study = study.sample(frac=1)
    study['sp'] = np.arange(study.shape[0])+1
    study['duration'].iloc[0] = retention_delay
    test = study.copy()
    test['duration'] = 0.5
    test['duration'].iloc[0] = retention_delay
    test['procedure'] = 'test'
    dat = pd.concat([study, test])
    dat['trial'] = np.arange(dat.shape[0]) + 1
    dat['list'] = 'context' + str(listn)
    dat = dat.rename(columns = {'word': 'stim1'})
    dat['isi'] = isi
    if (isi == 10):
        dat['reduce_wm'] = False
    else:
        dat['reduce_wm'] = False
    return(dat)

def gen_trials(n_subjects, test_type):
    """ generates trials to be used for modelling for n_subjects*10 """
    trials = pd.DataFrame()
    for subject in np.arange(n_subjects):
        listids = [1,2,3,4]
        random.shuffle(listids)
        hf10 = gen_list(12, 'hf', 40, 1000, 2, 10, 10, listids[0])
        lf10 = gen_list(12, 'lf', 0, 1, 2, 10, 10, listids[1])
        hf0 = gen_list(12, 'hf', 40, 1000, 2, 0, 10, listids[2])
        lf0 = gen_list(12, 'lf', 0, 1, 2, 0, 10, listids[3])
        dat = pd.concat([hf10, lf10, hf0, lf0])
        dat = dat.sort_values(by=['list','trial'])
        dat['subject'] = subject
        dat['trial'] = np.arange(dat.shape[0]) +1
        dat['onset'] = dat['duration'].cumsum()
        trials = pd.concat([trials, dat])
        trials['testtype'] = test_type
    return(trials)
    
#%%
#### The trials were originally generated with this command, but it takes
#### several minutes, so just load the presaved dataframe
np.random.seed(1274589)
trials = gen_trials(10, 'free_recall')
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
             'w_recovery_rate': 0.654,
             'y': 0.8,
             'multiply_activation': True}


""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation11',
             'episodeLearning': 'learning_equation11',
             'linkLearning': 'learning_equation2',
             'conceptDecay': 'decay_power',
             'episodeDecay': 'decay_power',
             'linkDecay': 'decay_power'}
    
filter_cond = ('procedure == "test"')
stim_columns = ['stim1','list'] 
group_vars = ['freq','isi'] 
split_nets_by = ['subject','list']

trials['onset'] = trials['onset']/init_pars['SCALE']
#trials = sac.utils.select_subjects(trials, np.arange(10)+1)
stim_info = sac.utils.get_stim_info(trials, stim_columns, init_pars)
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

#%%
init_pars['w_recovery_rate'] = 0.65
for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.527, 0.356], results['epiA']) #free recall
#results['acc_pred'] = sac.fit_model_group.activation_to_response([0.714, 0.749], results['epiA']) #item recognition
#results.to_csv('output/mixed_proportion_simultated_results_itemrecognition.csv')


fit = results.query(filter_cond).groupby(group_vars)['epiA','epiB','epiS'].mean().reset_index()
(ggplot(aes('isi','epiA', color='freq', group='freq'), data=fit) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
data = pd.DataFrame(dict(freq = ['hf','hf','lf','lf'],
                         isi = [0,10,0,10],
                         acc = [0.56, 0.37, 0.45, 0.35],
                          procedure = 'test'))

(ggplot(aes('isi','acc', color='freq', group='freq', shape='type'), data=data) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
#init_pars['w_recovery_rate'] = 0.8
model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(0.65,0.66,0.005)
                        },              
                        equations=equations, 
                        group_vars=['freq','isi'], 
                        target_obs=data,
                        fit_filter='procedure == "test"',
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
pred = pred.melt(id_vars = ['freq','isi'], value_vars=['acc_pred','acc'])

(ggplot(aes('isi','value',color='freq', shape='variable', group='freq+variable', linetype='variable'), data=pred)+
 geom_point()+
 geom_line()+
# coord_cartesian(ylim=[0,1]) +
 theme_classic())

#%%
#act_pars = [model.opt_pars['epiA_act_theta'], model.opt_pars['epiA_act_sd']]
#model.results_full['acc_pred'] = sac.fit_model_group.activation_to_response(act_pars,
#                  model.results_full['epiA'])

results1 = pd.merge(model.results, data, how='left')
results1.to_csv('output/final/gregg1980_sim.csv')
model.save_pars('gregg1980')

