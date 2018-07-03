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

import sac.main
import sac.utils
import sac.fit_model_group

from importlib import reload
reload(sac.equations)
reload(sac.main)
reload(sac.utils)
reload(sac.fit_model_group)

os.chdir('D:\\gdrive\\research\\projects\\122-sac-modelling\\')

subjects_to_fit = None
#subjects_to_fit = np.arange(10)
#subjects_to_fit = random.samp

""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1000.0,
             'W': 3,
             'contextAct': 1,
             'decaytype': 'power',
             'dl': -0.12,
             'dn': -0.18,
             'fan': 2,
             'p': 0.564,
             'p_additional': 0.715,
             'w_act': 2,
             'prior0': 0.4,
             'prior1': 0.2,
             'prior2': -0.1,
             'sem_theta': 1,
             'w_recovery_rate': 0.551,
             'y': 0.2,
             'free_recall_exp': 22}


""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation11',
             'episodeLearning': 'learning_equation11',
             'linkLearning': 'learning_equation2',
             'conceptDecay': 'decay_power',
             'episodeDecay': 'decay_power',
             'linkDecay': 'decay_power'}
    
filter_cond = ('procedure == "test" ' +
              'and cue_type_prioritem1 != "nan"')

stim_columns = ['stim1','stim2','list'] 
group_vars = ['cue_type','cue_type_consec_value'] 
split_nets_by = ['subject','list']

""" LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
allsubjects = sac.utils.load_trials('data/formatted_for_modeling/marevic2018.csv', 
                                    scale=init_pars['SCALE'],
                                    encoding='latin1')
allsubjects['testtype'] = 'nan'
allsubjects['testtype'][allsubjects['procedure'] == 'test'] = "cued_recall"
trials = sac.utils.select_subjects(allsubjects, subjects_to_fit)

# split each trial into two for base learning and then optional
study = trials[trials['procedure'] == 'study']
test = trials[trials['procedure'] == 'test']
test['trial'] = test['trial']+24
study = pd.concat([study,study])
study = study.sort_values(by=['subject','list','trial'])
study['trial'] = np.tile(np.arange(48)+1,int(study.shape[0]/48))
study['cue_type'][study['trial'] % 2 == 1] = 'base'
study['onset'] = (study['trial']-1) * 1.5
trials = pd.concat([study,test])
trials = trials.sort_values(by=['subject','list','procedure','trial'])

stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
stim_info['priorBase'] = 0.3
stim_info['priorFan'] = 0.6

#%%
nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)


#%%
#init_pars['p'] = 0.3
#init_pars['p_additional'] = 0.7
#init_pars['w_recovery_rate'] = 0.4

for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)

fit = results.query(filter_cond).groupby(group_vars)['epiA'].mean().reset_index()

(ggplot(aes('cue_type_consec_value','epiA', color='cue_type'), data=fit) +
     geom_point() +
     geom_line())

#%%
(ggplot(aes('cue_type_consec_value','wmSpent', color='cue_type'), data=results.query('procedure == "study"')) +
     stat_summary() +
     stat_summary(geom='line'))

#%%
init_pars['p'] = 0.553
init_pars['p_additional'] = 0.553
init_pars['w_recovery_rate'] = 0.5265

for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)   

fit = (results.melt(id_vars=['procedure','cue_type','epiA'], 
                    value_vars=['cue_type_prioritem1',
                                'cue_type_prioritem2',
                                'cue_type_prioritem3',
                                'cue_type_prioritem4']))
fit = fit.query('procedure == "test" & (value == "tbr" | value == "tbf")').groupby(['variable','value','cue_type'])['epiA'].mean().reset_index()
fit['Data'] = 'predicted'

(ggplot(aes(x='variable',y='epiA', shape='value', color='cue_type', linetype='Data', group='Data+cue_type+value'), 
        data=fit) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())

#%%

model = sac.fit_model_group.Model(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['free_recall_exp'],
                        par_ranges={'w_recovery_rate': slice(0.3,0.80,0.005),
                                                  'p': slice(0.3,0.8,0.02),
                                      'p_additional': slice(0.3,0.8,0.02),
                                      'free_recall_exp': slice(10,20,1)
                        },              
                        equations=equations, 
                        group_vars=group_vars, 
                        target_obs=allsubjects, 
                        resp_vars=['cued_recall_acc'], 
                        fit_filter=filter_cond,
                        verbose=True,
                        reject_related=False)

#%%
model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)

#%%

model.fit_with_final_parameters()
results = model.results
#results.to_csv('output/marevic2018_results_fit.csv')
#pd.DataFrame([model.opt_pars]).to_csv('output/marevic_exp1_parsfit_fmin_180323.csv')

# <codecell> OVERALL FIT OF THE MODEL
#fit = model.pred_smry.copy()
fit = results.query(filter_cond).groupby(group_vars)['cued_recall_acc_pred'].mean().reset_index()
fit['Data'] = 'predicted'
fit['cued_recall_acc'] = fit['cued_recall_acc_pred']
obs = allsubjects.query(filter_cond).groupby(group_vars)['cued_recall_acc'].mean().reset_index()
obs['Data'] = 'observed'
res = pd.concat([fit,obs])

f1 = (ggplot(aes(x='cue_type_consec_value', y='cued_recall_acc', color='cue_type', shape='Data', 
                 linetype='Data', group='cue_type+Data'), data=res) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic())
f1.draw()

#%%
fit = (results.melt(id_vars=['cued_recall_acc_pred','procedure','cue_type','epiA','cued_recall_acc'], 
                    value_vars=['cue_type_prioritem1',
                                'cue_type_prioritem2',
                                'cue_type_prioritem3',
                                'cue_type_prioritem4']))
fit = fit.query('procedure == "test" & (value == "tbr" | value == "tbf")').groupby(['variable','value','cue_type'])['cued_recall_acc_pred','cued_recall_acc'].mean().reset_index()
fit = fit.melt(id_vars=['value','variable','cue_type'],
               value_vars = ['cued_recall_acc','cued_recall_acc_pred'],
               var_name='Data', value_name='acc')
#    
(ggplot(aes(x='variable',y='acc', shape='value', color='cue_type', linetype='Data', group='Data+cue_type+value'), 
        data=fit) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())


#%%
#init_pars['p'] = 0.575990
#init_pars['p_additional'] = 0.580103
#init_pars['w_recovery_rate'] = 0.542241

#init_pars['p'] = 0.553
#init_pars['p_additional'] = 0.553
#init_pars['w_recovery_rate'] = 0.5265

model = sac.fit_model_group.MarevicModel(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate','p','p_additional'],
                        par_ranges={'w_recovery_rate': slice(0.3,0.8,0.01),
                                                  'p': slice(0.3,0.8,0.02),
                                      'p_additional': slice(0.3,0.8,0.02),
                        },              
                        equations=equations, 
                        group_vars=group_vars, 
                        target_obs=allsubjects, 
                        resp_vars=['cued_recall_acc'], 
                        fit_filter=filter_cond,
                        verbose=True,
                        reject_related=False)
#model.estimate_parameters(order='joint', func_type='fmin', rep=1, ftol=0.001, duration=True)
model.fit_with_final_parameters()
results = model.results
#results.to_csv('output/marevic2017_results_fit_cued_recall.csv')
#pd.DataFrame([model.opt_pars]).to_csv('output/marevic_exp1_parsfit_fmin_180323_cued_recall.csv')

#%%
model.pred_smry['acc'] = model.obs_data_smry.values
fit = model.pred_smry.melt(id_vars=['value','variable','cue_type'],
               value_vars = ['acc','cued_recall_acc_pred'],
               var_name='Data', value_name='acc')
(ggplot(aes(x='variable',y='acc', shape='value', color='cue_type', linetype='Data', group='Data+cue_type+value'), 
        data=fit) +
     stat_summary(geom='point') +
     stat_summary(geom='line')+
     theme_classic())

# <codecell> OVERALL FIT OF THE MODEL
#fit = model.pred_smry.copy()
#model.fit_with_final_parameters()
fit = results.query(filter_cond).groupby(group_vars)['cued_recall_acc','cued_recall_acc_pred'].mean().reset_index()
fit = fit.melt(id_vars=['cue_type','cue_type_consec_value'],
               value_vars = ['cued_recall_acc','cued_recall_acc_pred'],
               var_name='Data', value_name='acc')

f1 = (ggplot(aes(x='cue_type_consec_value', y='acc', color='cue_type', shape='Data', 
                 linetype='Data', group='cue_type+Data'), data=fit) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic())
f1.draw()


# <codecell> OVERALL FIT OF THE MODEL
#fit = model.pred_smry.copy()
#model.fit_with_final_parameters()
fit = results.query(filter_cond).groupby(['cue_type','cue_type_prioritem1'])['cued_recall_acc', 'cued_recall_acc_pred'].mean().reset_index()
fit = fit.melt(id_vars=['cue_type','cue_type_prioritem1'],
               value_vars = ['cued_recall_acc','cued_recall_acc_pred'],
               var_name='Data', value_name='acc')

f1 = (ggplot(aes(x='cue_type', y='acc', color='cue_type_prioritem1', shape='Data', 
                 linetype='Data', group='cue_type_prioritem1+Data'), data=fit) +
     stat_summary(size=0.8) +
     stat_summary(geom='line') +
     theme_classic())
f1.draw()