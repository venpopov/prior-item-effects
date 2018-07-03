import sys
import pandas as pd
import numpy as np
import random
import scipy as sp
import sac.utils
import sac.main
import sac.fit_model
import importlib
import matplotlib.pyplot as plt
from plotnine import *
#from pandas_ply import install_ply, X, sym_call
#install_ply(pd)
importlib.reload(sac.equations)
importlib.reload(sac.main)
importlib.reload(sac.utils)
importlib.reload(sac.fit_model)



#%%
""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1.0,
               'W': 3,
               'contextAct': 1,
               'decaytype': 'power',
               'dl': -0.12,
               'dn': -0.18,
               'fan': 2,
               'p': 0.8,
               'prior0': 0.4,
               'prior1': 0.2,
               'prior2': -0.1,
               'sem_theta': 1,
               'w_recovery_rate': 0.97,
               'y': 0.2,
               'w_act': 2}

""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation10',
              'episodeLearning': 'learning_equation10',
              'linkLearning': 'learning_equation2',
              'conceptDecay': 'decay_power',
              'episodeDecay': 'decay_power',
              'linkDecay': 'decay_power'}


""" LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
allsubjects = sac.utils.load_trials('data/cox2018.csv', scale=init_pars['SCALE'])
# divide free recall accuracy by 2 because right now it's coded as 2 for full recall
allsubjects['acc'][allsubjects['condition'] == 'free recall'] *= 0.5
trials = sac.utils.select_subjects(allsubjects, np.arange(30))
#allsubjects = allsubjects.query('source != "fake"')

stim_columns = ['list', 'stim1','stim2']
stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)
#stim_info = (stim_info.join(allsubjects[allsubjects['procedure'] == 'study'][['stim1','freq']].
#                            drop_duplicates().
#                            set_index('stim1'), 
#                            lsuffix='_'))
#stim_info['priorBase'][stim_info['freq'] == 'low'] = 0.2
#stim_info['priorBase'][stim_info['freq'] == 'high'] = 0.4
#stim_info['priorBase'][stim_info['freq'].isnull()] = 0.3
#stim_info['priorFan'][stim_info['freq'] == 'low'] = 0.4
#stim_info['priorFan'][stim_info['freq'] == 'high'] = 0.8
#stim_info['priorFan'][stim_info['freq'].isnull()] = 0.6

split_nets_by = ['subject','list']
group_vars = ['condition','freq_prioritem1']
filter_cond = '(procedure=="test") and (freq_prioritem1 != "nan") and studied == 1'


nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

for net in nets.values():
    net.run_trials(init_pars, equations)
    
results = sac.utils.extract_results(nets)

fit = results.query(filter_cond).groupby(['condition','freq_prioritem1'])['epiB','epiA','stim1.semB','acc','wmSpent'].mean().reset_index()
res = allsubjects.query(filter_cond).groupby(['condition','freq_prioritem1'])['acc'].mean().reset_index()

#%%
#fit = results.query('procedure == "study"').groupby(group_vars+['freq1'])['epiS','wmSpent'].mean().reset_index()
fit = results.query(filter_cond).groupby(['condition','freq_prioritem1'])['epiB','epiA','stim1.semB','acc','wmSpent'].mean().reset_index()

(ggplot(aes('freq_prioritem1','epiA', color='condition'), data=fit) + 
 geom_point() +
 geom_smooth(se=False,method='lm') +
 facet_wrap('condition', scales='free'))

# %%
def plot_func(w):
    init_pars['w_recovery_rate'] = w
    for net in nets.values():
        net.run_trials(init_pars, equations)
    results = sac.utils.extract_results(nets)
    fit = results.query('procedure == "study"').groupby(group_vars+['freq1'])['epiS','wmSpent'].mean().reset_index()
#    res = (results.melt(id_vars=['epiA','procedure','freq'], 
#                        value_vars=['freq_prioritem1',
#                                    'freq_prioritem2',
#                                    'freq_prioritem3',
#                                    'freq_prioritem4']))
    f1 = (ggplot(aes('freq1','epiS'), data=fit) + stat_summary() + geom_smooth(se=False,method='lm'))
    return(f1)

#%%
import sac.fit_model_group
importlib.reload(sac.fit_model_group)
model = sac.fit_model_group.CoxModel(nets=nets, 
                        init_pars=init_pars, 
                        par_names=['w_recovery_rate'],
                        par_ranges={'w_recovery_rate': slice(0.97,0.98,0.01),
                                                  'W': slice(1.4,5,0.1),
                        },              
                        equations=equations, 
                        group_vars=['condition','freq_prioritem1'], 
                        target_obs=allsubjects,
                        fit_filter=filter_cond,
                        resp_vars=['acc'],
                        verbose=True,
                        reject_related=False)

#model.estimate_parameters(order='sequential', func_type='brute', rep=1, ftol=0.001, duration=True)
model.fit_with_final_parameters()
results = model.results
results.to_csv('output/cox2018_results_fit.csv')


#%%
fit = results.query(filter_cond).groupby(group_vars)['acc_pred'].mean().reset_index()
fit['acc'] = fit['acc_pred']
fit['Data'] = 'predicted'
res = allsubjects.query(filter_cond).groupby(group_vars)['acc'].mean().reset_index()
res['Data'] = 'observed'
res = pd.concat([fit,res])

(ggplot(aes('freq_prioritem1','acc', color='condition', shape='Data', linetype='Data'), data=res) + 
 geom_point() +
 geom_smooth(se=False,method='lm') +
 facet_wrap('condition', scales='free'))

