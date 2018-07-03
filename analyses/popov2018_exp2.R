# -------------------------------------------------------------------
# GOAL: Analyze data from Popov, Marevic, Rummel & Reder (2018), Exp 2.
# AUTHOR: Ven Popov
# DATE: 06-June-2018
# -------------------------------------------------------------------

#############################################################################
# SETUP
#############################################################################

rm(list=ls())
library(tidyverse)
library(here)
library(lme4)
library(brms)
library(cowplot)
setwd(here())
source('analyses/prior_item_functions.R')
theme_set(theme_classic(base_size=9))

#############################################################################
# DATA
#############################################################################

# load and preprocess data
raw <- read.csv('data/popov2018_exp2.csv')
names(raw) <- c('subject','group','cue','condition','list','stim','free_recall_pair','free_recall_single','free_recall_acc','cued_recall_acc')
dat <- raw %>% 
  mutate(cue = ifelse(cue == "**VVV**", 'TBF','TBR')) %>% 
  group_by(subject, list) %>% 
  mutate(trial = 1:length(subject)) %>% 
  prior_item_analysis('cue', 'cue', 'TBF', 3, 4) %>% # get the cue type for the preceding study item
  select(-cue_prioritem)



dat <- dat %>% 
  mutate(free_recall_acc = ifelse(is.na(free_recall_acc), 0, free_recall_acc)) %>% 
  mutate(condition = recode(condition, Control='Control',Rehearsal='Reh',Attention='Att',RehearsalAttention='Reh+Att'),
         condition = as.factor(condition),
         condition = factor(condition, levels=c('Control','Att', 'Reh','Reh+Att'))) %>% 
  separate(stim, c('stim1','stim2'), sep=' - ') %>% 
  mutate(stim1 = tolower(stim1))

# load sac fit
sac_fit <- read.csv('output/popov2018_sac_model_fit.csv')
names(sac_fit)[grepl('type', names(sac_fit))] <- gsub('_type_','',names(sac_fit)[grepl('type', names(sac_fit))])
sac_fit <- select(sac_fit,subject, list, stim1, cued_recall_acc_pred, free_recall_acc_pred)
sac_fit$list <- gsub('context','', sac_fit$list) %>% toupper()
dat <- left_join(dat, sac_fit)
dat <- filter(dat, !is.na(cue_prioritem1), cue != 'TBF')

# save processed data
# write.csv(dat, 'data/exp2_processed.csv', row.names=FALSE)


# -------------------------------------------------------------------
# MODEL FIT STATISTIC
# -------------------------------------------------------------------

fit <- dat %>% 
  group_by(cue_consec_lab) %>% 
  summarise(cued_recall_acc=mean(cued_recall_acc), 
            cued_recall_acc_pred=mean(cued_recall_acc_pred, na.rm=T), 
            free_recall_acc=mean(free_recall_acc), 
            free_recall_acc_pred=mean(free_recall_acc_pred, na.rm=T))

sqrt(mean((fit$cued_recall_acc-fit$cued_recall_acc_pred)^2))
sqrt(mean((fit$free_recall_acc-fit$free_recall_acc_pred)^2))
cor(fit$cued_recall_acc,fit$cued_recall_acc_pred)^2
cor(fit$free_recall_acc,fit$free_recall_acc_pred)^2

# -------------------------------------------------------------------
# PLOTS WITH SAC FITS
# -------------------------------------------------------------------

# basic cued recall
(f1 <- dat %>% 
   group_by(subject, condition, cue_prioritem1) %>%
   summarise(acc = mean(cued_recall_acc),
             acc_pred = mean(cued_recall_acc_pred)) %>%
   Rmisc::normDataWithin('subject', 'acc') %>% 
   # gather(data,acc1,accNormed,acc_pred) %>% 
   # mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
   ggplot(aes(condition, accNormed, shape=cue_prioritem1, group=cue_prioritem1)) +
   stat_summary(geom='pointrange') +
   stat_summary(geom='line') +
   coord_cartesian(ylim=c(0.1,0.6)) +
   scale_x_discrete('Dual-task condition\n') +
   scale_y_continuous('a) Cued recall accuracy') +
   scale_linetype_discrete('Data vs model') +
   scale_shape_discrete("Previous item type"))

# basic free recall
(f2 <- dat %>% 
    group_by(subject, condition, cue_prioritem1) %>%
    summarise(acc = mean(free_recall_acc),
              acc_pred = mean(free_recall_acc_pred)) %>%
    Rmisc::normDataWithin('subject', 'acc') %>% 
    # gather(data,acc1,accNormed,acc_pred) %>% 
    # mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
    ggplot(aes(condition, accNormed, shape=cue_prioritem1, group=cue_prioritem1)) +
    stat_summary(geom='pointrange') +
    stat_summary(geom='line') +
    coord_cartesian(ylim=c(0.1,0.6)) +
    scale_x_discrete('Dual-task condition\n') +
    scale_y_continuous('d) Free recall accuracy') +
    scale_linetype_discrete('Data vs model') +
    scale_shape_discrete("Previous item type"))

# cumulative effect, cued recall
(f3 <- dat %>% 
    group_by(subject, cue_consec_lab, cue_consec_lab, cue_prioritem1, cue) %>%
    summarise(acc = mean(cued_recall_acc),
              acc_pred = mean(cued_recall_acc_pred)) %>%
    Rmisc::normDataWithin('subject', 'acc') %>% 
    gather(data,acc1,accNormed,acc_pred) %>% 
    mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
    ggplot(aes(cue_consec_lab, acc1, group=data, linetype=data, fill=data)) +
    stat_summary(geom='pointrange', shape=21) +
    stat_summary(geom='line') +
    scale_x_discrete('# of immediately preceding\n TBF or TBR items') +
    scale_linetype_discrete('Data vs model') +
    scale_fill_manual('Data vs model', values=c('black','white')) +
    scale_y_continuous('b) Cued recall accuracy'))

# cumulative effect, free recall
(f4 <- dat %>% 
    group_by(subject, cue_consec_lab, cue_consec_lab, cue_prioritem1, cue) %>%
    summarise(acc = mean(free_recall_acc),
              acc_pred = mean(free_recall_acc_pred)) %>%
    Rmisc::normDataWithin('subject', 'acc') %>% 
    gather(data,acc1,accNormed,acc_pred) %>% 
    mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
    ggplot(aes(cue_consec_lab, acc1, group=data, linetype=data, fill=data)) +
    stat_summary(geom='pointrange', shape=21) +
    stat_summary(geom='line') +
    scale_x_discrete('# of immediately preceding\n TBF or TBR items') +
    scale_linetype_discrete('Data vs model') +
    scale_fill_manual('Data vs model', values=c('black','white')) +
    scale_y_continuous('e) Free recall accuracy'))

(f5 <- dat %>% 
    gather(lag, cue_prioritem, cue_prioritem1, cue_prioritem2, cue_prioritem3, cue_prioritem4) %>% 
    filter(!is.na(cue_prioritem)) %>% 
    group_by(subject, lag, cue_prioritem, cue) %>%
    summarise(acc = mean(cued_recall_acc, na.rm=T),
              acc_pred = mean(cued_recall_acc_pred, na.rm=T)) %>%
    Rmisc::normDataWithin('subject', 'acc') %>% 
    gather(data,acc1,accNormed,acc_pred) %>% 
    mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
    Rmisc::normDataWithin('subject', 'acc') %>% 
    ggplot(aes(lag, acc1, shape=cue_prioritem, group=interaction(cue_prioritem, data), linetype=data, fill=data)) +
    stat_summary(geom='pointrange') +
    stat_summary(geom='line') +
    scale_x_discrete('Lag between current and\nprior study item', labels=1:4) +
    scale_y_continuous('c) Cued recall accuracy') +
    scale_shape_manual('Previous item type', values=c(21,24))  +
    scale_linetype_discrete('Data vs model') +
    scale_fill_manual('Data vs model', values=c('black','white')))

(f6 <- dat %>% 
    gather(lag, cue_prioritem, cue_prioritem1, cue_prioritem2, cue_prioritem3, cue_prioritem4) %>% 
    filter(!is.na(cue_prioritem)) %>% 
    group_by(subject, lag, cue_prioritem, cue) %>%
    summarise(acc = mean(free_recall_acc, na.rm=T),
              acc_pred = mean(free_recall_acc_pred, na.rm=T)) %>%
    Rmisc::normDataWithin('subject', 'acc') %>% 
    gather(data,acc1,accNormed,acc_pred) %>% 
    mutate(data = recode(data, accNormed = 'Observed', acc_pred = 'SAC model fit')) %>% 
    Rmisc::normDataWithin('subject', 'acc') %>% 
    ggplot(aes(lag, acc1, shape=cue_prioritem, group=interaction(cue_prioritem, data), linetype=data, fill=data)) +
    stat_summary(geom='pointrange') +
    stat_summary(geom='line') +
    scale_x_discrete('Lag between current and\nprior study item', labels=1:4) +
    scale_y_continuous('f) Free recall accuracy') +
    scale_shape_manual('Previous item type', values=c(21,24))  +
    scale_linetype_discrete('Data vs model') +
    scale_fill_manual('Data vs model', values=c('black','white')))

legend <- g_legend(f5 + theme(legend.position='right'))
no_legend <- theme(legend.position = 'none')
(all_plots <- plot_grid(f1+no_legend, f2+no_legend, NULL, NULL,
                        f3+no_legend, f4+no_legend, NULL, legend,
                        f5+no_legend, f6+no_legend, NULL, NULL,
                        nrow = 3,
                        rel_widths = c(0.375,0.375, 0.05, 0.2)))

ggsave('figures/exp2_results_fit.tiff', all_plots, width=6.75, height=8, units='in', compression='lzw')


#############################################################################
# ANALYSES
#
# these take a long time, so they can be loaded directly from the pre-saved 
# RData objects that are stored on the OSF repository (https://osf.io/5qd94/files/)
# Download the .RData files and store them in the output folder
#############################################################################

load('output/exp2_cued_recall_regressions.RData')
load('output/exp2_cued_recall_regressions_consec_value.RData')
load('output/exp2_cued_recall_regressions_lag.RData')

# # run bayesian multilevel logistic regression of cued recall as a function of current and preceding cue type
# ml0 <- brm(cued_recall_acc ~ 1 + (condition + cue_prioritem1||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# ml1 <- brm(cued_recall_acc ~ cue_prioritem1 + (condition + cue_prioritem1||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# ml2 <- brm(cued_recall_acc ~ cue_prioritem1 + condition + (condition + cue_prioritem1||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# ml3 <- brm(cued_recall_acc ~ cue_prioritem1 * condition + (condition + cue_prioritem1||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# 
# 
# bf10 <- bayes_factor(ml1, ml0)
# bf21 <- bayes_factor(ml2, ml1)
# bf32 <- bayes_factor(ml3, ml2)
# save(ml0, ml1, ml2, ml3, bf10, bf21, bf32, file='output/exp2_cued_recall_regressions.RData')
# 
# 
# # run bayesian multilevel logistic regression of cued recall as a function of number of consecutive cue type
# mla_0 <- brm(cued_recall_acc ~ 1 + (condition + cue_consec_value||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# mla_1 <- brm(cued_recall_acc ~ cue_consec_value + (condition + cue_consec_value||subject) + (1||stim1), data=dat, family=bernoulli(), save_all_pars = TRUE, iter = 10000)
# 
# bfa_10 <- bayes_factor(mla_1, mla_0)
# save(mla_0, mla_1, bfa_10, file='output/exp2_cued_recall_regressions_consec_value.RData')
# 
# # run bayesian multilevel logistic regression of cued recall as a function of preceding cue type and lag
# mlc_1 <- brm(cued_recall_acc ~ cue_prioritem2 + cue_prioritem3 + cue_prioritem4 + (condition + cue_prioritem1||subject) + (1||stim1), data=filter(dat, !is.na(cue_prioritem4)), family=bernoulli(), save_all_pars = TRUE, iter = 10000, control = list(adapt_delta = 0.98))
# mlc_2 <- brm(cued_recall_acc ~ cue_prioritem1 + cue_prioritem2 + cue_prioritem4 + (condition + cue_prioritem1||subject) + (1||stim1), data=filter(dat, !is.na(cue_prioritem4)), family=bernoulli(), save_all_pars = TRUE, iter = 10000, control = list(adapt_delta = 0.98))
# mlc_3 <- brm(cued_recall_acc ~ cue_prioritem1 + cue_prioritem3 + cue_prioritem4 + (condition + cue_prioritem1||subject) + (1||stim1), data=filter(dat, !is.na(cue_prioritem4)), family=bernoulli(), save_all_pars = TRUE, iter = 10000, control = list(adapt_delta = 0.98))
# mlc_4 <- brm(cued_recall_acc ~ cue_prioritem1 + cue_prioritem2 + cue_prioritem3 + (condition + cue_prioritem1||subject) + (1||stim1), data=filter(dat, !is.na(cue_prioritem4)), family=bernoulli(), save_all_pars = TRUE, iter = 10000, control = list(adapt_delta = 0.98))
# mlc_5 <- brm(cued_recall_acc ~ cue_prioritem1 + cue_prioritem2 + cue_prioritem3 + cue_prioritem4 + (condition + cue_prioritem1||subject) + (1||stim1), data=filter(dat, !is.na(cue_prioritem4)), family=bernoulli(), save_all_pars = TRUE, iter = 10000, control = list(adapt_delta = 0.98))
# 
# bfc_51 <- bayes_factor(mlc_5,mlc_1)
# bfc_52 <- bayes_factor(mlc_5,mlc_2)
# bfc_53 <- bayes_factor(mlc_5,mlc_3)
# bfc_54 <- bayes_factor(mlc_5,mlc_4)
# 
# hypothesis(mlc_5, 'cue_prioritem1TBR < cue_prioritem2TBR')
# hypothesis(mlc_5, 'cue_prioritem2TBR < cue_prioritem3TBR')
# hypothesis(mlc_5, 'cue_prioritem3TBR < cue_prioritem4TBR')
# 
# save(mlc_1, mlc_2, mlc_3, mlc_4, mlc_5, bfc_51, bfc_52, bfc_53, bfc_54, file='output/exp2_cued_recall_regressions_lag.RData')
