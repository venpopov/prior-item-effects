import sac.main
import sac.utils
import sac.fit_model
import importlib
import numpy as np
from matplotlib import pyplot as plt
from plotnine import * 
import pandas as pd
import scipy.stats
importlib.reload(sac.equations)
importlib.reload(sac.main)
importlib.reload(sac.utils)
importlib.reload(sac.fit_model)
import warnings
from pdb import set_trace as bp
warnings.filterwarnings("ignore")

""" DEFINE CONSTANTS TO USE IN THE MODEL """
init_pars = {'SCALE': 1000.0,  # time scale to transform into seconds
             'dn': -0.18,           # node pwer decay parameter
             'dl': -0.12,           # link power decay parameter
             'p': 0.8,            # learning paramter
             'y': 0.2,            # current activation decay parameter
             'sem_theta': 1,
             'contextAct': 1,
             'prior0': 0.4,
             'prior1': 0.2,
             'prior2': -0.1,
             'fan': 2,
             'decaytype': 'power',
             'W': 5.5,
             'w_recovery_rate': 10,
             'w_act': 1,
             }


""" SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
equations = {'conceptLearning': 'learning_equation4',
              'episodeLearning': 'learning_equation4',
              'linkLearning': 'learning_equation2',
              'conceptDecay': 'decay_power',
              'episodeDecay': 'decay_power',
              'linkDecay': 'decay_power'}

""" LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
allsubjects = sac.utils.load_trials('data/formatted_for_modeling/reder2000_exp1.csv', scale=init_pars['SCALE'])
allsubjects['rem'] = (allsubjects.resp == 'rem').astype(int)
allsubjects['know'] = (allsubjects.resp == 'know').astype(int)
allsubjects['stim1'] = allsubjects['stim']
allsubjects = allsubjects[allsubjects['stim1']!='gayety']
trials = sac.utils.select_subjects(allsubjects, [13])


stim_columns = ['list', 'stim1']
stim_info = sac.utils.get_stim_info(allsubjects, stim_columns, init_pars)


split_nets_by = ['subject']
group_vars = ['freq','rep']
filter_cond = 'subject > 0'

nets = sac.utils.init_networks(trials, split_nets_by, stim_columns, 
                               init_pars, stim_info, duration=True)

model = sac.fit_model.RemKnowModel(nets=nets, 
                        init_pars=init_pars, 
                        shared_par_names=None,
#                        subject_par_names=['p_epi','p_sem','dn_sem','dn_epi'],
                        subject_par_names=None,
                        par_ranges={
                        },              
                        equations=equations, 
                        group_vars=group_vars, 
                        target_obs=allsubjects,
                        fit_filter=filter_cond,
                        verbose=True)

model.estimate_parameters(order='joint', func_type='fmin', rep=2, ftol=0.01, duration=True)
results = sac.utils.extract_results(nets)

smry = sac.utils.summarise_obs_and_predictions(results, 
                                               allsubjects, 
                                               filter_cond, 
                                               group_vars, 
                                               ['rem','know'],
                                               ['rem_pred','know_pred'])

f1 = (ggplot(smry, aes('rep','value',color='variable')) +
     geom_point() +
     geom_line() +
     facet_wrap('~freq'))
f1.draw()

#%%
(ggplot(results, aes('rep','epiA',color='freq')) +
     stat_summary() +
     stat_summary(geom='line')).draw()