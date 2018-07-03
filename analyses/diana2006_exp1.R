# Reanalyzes data from the following experiment: 
#   Diana, R. A., & Reder, L. M. (2006). The low-frequency encoding disadvantage: 
#   Word frequency affects processing demands. Journal of Experimental Psychology: 
#   Learning, Memory, and Cognition, 32(4), 805.
# SCRIPT AUTHOR: Ven Popov

#############################################################################
# SETUP
#############################################################################

rm(list=ls())
library(tidyverse)
library(here)
library(lme4)
library(cowplot)
source(here('analyses/prior_item_functions.R'))
theme_set(theme_classic(base_size=9))


#############################################################################
# PRIOR ITEM ANALYSIS
#############################################################################
# split into study and test, extract info about prior item freq during study
# then join with test

dat <- read.csv(here('data/diana2006_exp1.csv'))
dat$freq <- tolower(dat$freq)

study <- dat %>% 
  filter(procedure == 'study') %>% 
  select(subject, stim1, stim2, freq) %>% 
  group_by(subject) %>% 
  prior_item_analysis('freq','freq','low',4, 3)

test <- dat %>% 
  filter(procedure == 'test') %>% 
  select(subject, stim1, rt, trial,resp, rem, fam, new, acc)

dat <- study %>% 
  ungroup() %>% 
  left_join(test, by=c('subject','stim1')) %>% 
  filter(!is.na(freq_prioritem1),
         !(subject %in% c(9,20)))  # remove subjects with below chance performance

# load modelling results
# sac_fit <- read.csv('output/diana2006_model_fit.csv')
sac_fit <- read.csv('output/diana2006_sac_model_fit.csv')


#############################################################################
# PLOTS
#############################################################################

# accuracy by freq and freq_prioritem
(f1 <- dat %>% 
  ggplot(aes(freq, acc, fill=freq_prioritem, group=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the current item\n') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='Hit rate') +
  coord_cartesian(ylim=c(0.5,1.05)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)))
ggsave(here('figures/diana2006_acc_freq_freqprioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')


# cumulative effect of freq_consec
(f2 <- dat %>% 
  filter(abs(freq_consec_value) <= 4) %>%
  ggplot(aes(freq_consec_lab, acc, group=1)) +
  stat_summary(fun.data = mean_se, geom="pointrange") +
  stat_smooth(method='lm', se=FALSE, color='black') +
  # stat_summary(fun.data = mean_se, geom="line") +  
  scale_x_discrete(name='Number of consecutive LOW or HIGH\n frequency preceding items', 
                   labels=c('LF4','LF3','LF2','LF1','HF1','HF2','HF3','HF4')) +
  scale_color_discrete(name='Frequency of the current item') +
  scale_y_continuous(name='Hit rate') +
  theme_classic() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0)))
ggsave(here('figures/diana2006_acc_freqpriorconsec.tiff'), width=3.5, height=3.5, units='in', compression='lzw')

# cumulative effect of freq_consec interaction with current freq
dat %>% 
  filter(abs(freq_consec_value) <= 4) %>%
  ggplot(aes(freq_consec_lab, acc, color=freq, group=freq)) +
  stat_summary(fun.data = mean_se, geom="pointrange") +
  # stat_smooth(method='lm', se=FALSE) +
  stat_summary(fun.data = mean_se, geom="line") +  
  scale_x_discrete(name='Number of consecutive LOW or HIGH\n frequency preceding items', 
                   labels=c('LF4','LF3','LF2','LF1','HF1','HF2','HF3','HF4')) +
  scale_color_discrete(name='Frequency of the current item') +
  scale_y_continuous(name='Hit rate') +
  theme_classic() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0))

# effect of lag
(f3 <- dat %>% 
  filter(!is.na(freq_prioritem3), freq=='low') %>% 
  gather(lag, freq_prior, freq_prioritem1, freq_prioritem2, freq_prioritem3) %>% 
  ggplot(aes(lag, acc, fill=freq_prior, group=freq_prior)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Prior item lag\n', labels=c('Lag1','Lag2','Lag3')) +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='Hit rate') +
  coord_cartesian(ylim=c(0.5,1.05)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)))
ggsave(here('figures/diana2006_acc_freqprior_lag.tiff'), width=3.5, height=3.5, units='in', compression='lzw')

# combine all figures
legend <- g_legend(f1 + theme(legend.position = 'bottom'))
(f_all <- plot_grid(f1 + theme(legend.position = 'None') + annotate('text', x=1, y=1, label='a)'), 
                    f3 + theme(legend.position = 'None') + annotate('text', x=1, y=1, label='b)'), 
                    f2 + theme(legend.position = 'None') + annotate('text', x=2, y=0.885, label='c)'), 
                 NULL, legend, NULL, 
                 nrow = 2,
                 rel_heights = c(5/6,1/6)))
ggsave(here('figures/diana2006_all_figures.tiff'), f_all, width=6.875, height=2.8, units='in', scale=1.4)


#############################################################################
# PLOTS WITH SAC FITS OVERLAYS
#############################################################################

# accuracy by freq and freq_prioritem
(f1 <- dat %>% 
   ggplot(aes(freq, acc, fill=freq_prioritem, group=freq_prioritem)) +
   stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
   stat_summary(fun.data = mean_se, geom="errorbar", width=0.2, position=position_dodge(0.3)) +
   scale_x_discrete(name='Frequency of the current item\n') +
   scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
   scale_y_continuous(name='Hit rate') +
   coord_cartesian(ylim=c(0.5,1.05)) +
   theme_classic() +
   theme(legend.position = c(1,1),
         legend.justification = c(1,1)) + 
  stat_summary(data=sac_fit, aes(y=old_pred), geom='point', position=position_dodge(0.3), size=2))

ggsave(here('figures/new/diana2006_acc_freq_freqprioritem_sac.tiff'), width=3.5, height=3.5, units='in', compression='lzw')


# cumulative effect of freq_consec
pred <- sac_fit %>% 
  group_by(freq, freq_consec_lab) %>% 
  summarise(old_pred = mean(old_pred, na.rm=T))

(f2 <- dat %>% 
    left_join(pred) %>% 
    gather(variable, value, acc, old_pred) %>% 
    mutate(freq = recode(freq, high = 'HF current item', low='LF current item')) %>% 
    mutate(variable = recode(variable, acc = 'Data', old_pred = 'SAC model fit')) %>% 
    ggplot(aes(freq_consec_value, value, shape=variable, linetype=variable, fill=variable)) +
    stat_summary(geom='line') +
    stat_summary(fun.data = mean_se, geom="point", size=2.5) +
    # stat_summary(fun.data = mean_se, geom="line") +  
    scale_x_continuous(name='Number of consecutive LOW or HIGH\n frequency preceding items', breaks=sort(unique(dat$freq_consec_value)),
                     labels=c('LF4','LF3','LF2','LF1','HF1','HF2','HF3','HF4')) +
    scale_y_continuous(name='Hit rate') +
    scale_shape_manual(name='', values=c(21,24)) +
    scale_fill_manual(name='', values=c('black','white')) +
    scale_linetype_discrete(name='') +
    theme_bw(base_size=10) +
    theme(legend.position = c(0.95,0.05),
          legend.justification = c(1,0),
          panel.spacing.x = unit(1, 'lines'),
          panel.grid = element_blank()) +
    facet_wrap(~freq)) 

ggsave(here('figures/new/diana2006_acc_freqpriorconsec_sac.emf'), width=6.5, height=3.6, units='in')

# effect of lag
pred <- sac_fit %>% 
  filter(freq == 'low') %>% 
  gather(lag, freq_prior, freq_prioritem1, freq_prioritem2, freq_prioritem3) %>% 
  filter(freq_prior != 'nan') %>% 
  group_by(lag, freq_prior) %>% 
  summarise(old_pred = mean(old_pred, na.rm=T))


(f3 <- dat %>% 
    filter(!is.na(freq_prioritem3), freq=='low') %>% 
    gather(lag, freq_prior, freq_prioritem1, freq_prioritem2, freq_prioritem3) %>% 
    ggplot(aes(lag, acc, fill=freq_prior, group=freq_prior)) +
    stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
    stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
    stat_summary(data=pred, aes(y=old_pred), geom='point', position=position_dodge(0.3), size=2) +
    scale_x_discrete(name='Prior item lag\n', labels=c('Lag1','Lag2','Lag3')) +
    scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
    scale_y_continuous(name='Hit rate') +
    coord_cartesian(ylim=c(0.5,1.05)) +
    theme_classic() +
    theme(legend.position = c(1,1),
          legend.justification = c(1,1)))
ggsave(here('figures/new/diana2006_acc_freqprior_lag_sac.tiff'), width=3.5, height=3.5, units='in', compression='lzw')

# combine all figures
legend <- g_legend(f1 + theme(legend.position = 'bottom'))
(f_all <- plot_grid(f1 + theme(legend.position = 'None'), 
                    f3))
ggsave(here('figures/new/diana2006_f1_sac.emf'), f_all, width=6.5, height=3.2, units='in')


#############################################################################
# ANALYSES
#############################################################################

# mixed effects logistic regression of acc ~ freq_prioritem
ml0 <- glmer(acc ~  freq + (freq+freq_prioritem|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml1 <- glmer(acc ~  freq + freq_prioritem + (freq+freq_prioritem|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml2 <- glmer(acc ~  freq*freq_prioritem + (freq+freq_prioritem|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
sink(here('output/diana2006_lmer1.txt'))
summary(ml2)
cat('\n')
anova(ml0, ml1, ml2)
sink()


# mixed effects logistic regression of acc ~ freq_consec_value
ml3.0 <- glmer(acc ~  freq + (0+freq+freq:freq_consec_value|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml3 <- glmer(acc ~  freq + freq_consec_value + (0+freq+freq:freq_consec_value|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml4 <- glmer(acc ~  freq*freq_consec_value + (0+freq+freq:freq_consec_value|subject) + (0+freq|stim1), 
             data=dat, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
sink(here('output/diana2006_lmer2.txt'))
summary(ml3)
cat('\n')
anova(ml4, ml3, ml3.0)
sink()


# mixed effects logistic regression of acc ~ freq_prioritem + lag
dat1 <- dat %>% 
  gather(lag, freq_prior, freq_prioritem1, freq_prioritem2, freq_prioritem3) %>% 
  mutate(lag = as.numeric(as.factor(lag))) %>% 
  filter(!is.na(freq_prior))


ml6.1 <- glmer(acc ~  freq*freq_prior + (freq+freq_prior|subject) + (1|stim1), 
               data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml6.2 <- glmer(acc ~  freq*freq_prior + freq_prior*lag + freq*lag + (freq+freq_prior|subject) + (1|stim1), 
               data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml6.3 <- glmer(acc ~  freq*freq_prior*lag + (freq+freq_prior|subject) + (1|stim1), 
               data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
summary(ml6.3)
anova(ml6.1, ml6.2, ml6.3)

# post hoc z-test for lags
ml6.4 <- glmer(acc ~  0+as.factor(lag) + freq:freq_prior:as.factor(lag) + (freq_prior|subject) + (1|stim1), 
               data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
summary(ml6.4)
