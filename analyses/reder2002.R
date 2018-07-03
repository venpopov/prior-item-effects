# Reanalyzes data from the following experiment: 
#   Reder, L. M., Angstadt, P., Cary, M., Erickson, M. A., & Ayers, M. S. (2002). 
#   A reexamination of stimulus-frequency effects in recognition: Two mirrors for low- 
#   and high-frequency pseudowords. Journal of Experimental Psychology: Learning, 
#   Memory, and Cognition, 28(1), 138–152. 
# SCRIPT AUTHOR: Ven Popov

#############################################################################
# SETUP
#############################################################################

rm(list=ls())
library(tidyverse)
library(here)
library(lme4)
source(here('analyses/prior_item_functions.R'))


#############################################################################
# DATA
#############################################################################

dat <- read.csv(here('data/reder2002_exp1.csv'))
study <- dat %>% 
  filter(procedure == 'studyrecog') %>% 
  select(subject, session, freq,stim) %>% 
  mutate(freq = ifelse(freq == 'HF','HF','LF')) %>% 
  group_by(subject, session) %>% 
  prior_item_analysis('freq','freq','LF',3)

test <- dat %>% 
  filter(procedure == 'testrecog') %>% 
  select(-recall, -nrecall, -duration, -onset,-freq) %>% 
  left_join(ungroup(study), by=c('subject','session','stim')) %>% 
  filter(!is.na(freq_prioritem))
  

#############################################################################
# MAIN PLOTS
#############################################################################
# IES by freq and freq_prioritem
test %>%
  group_by(subject, freq, freq_prioritem) %>% 
  mutate(acc_mean = mean(acc),
         ies = rt/acc_mean) %>% 
  filter(acc == 1) %>% 
  group_by(subject) %>% 
  mutate(outlier = abs(rt-median(rt, na.rm=T))/mad(rt, na.rm=T)) %>%
  filter(outlier <= 4) %>%
  ggplot(aes(freq, ies, fill=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the recalled item') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='IES (RTs/accuracy)') +
  coord_cartesian(ylim=c(1200,2300)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1))
ggsave(here('figures/reder2002_ies_freq_freq_prioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')


# IES by freq_consec_value
test %>%
  group_by(subject, freq, freq_consec_value) %>% 
  mutate(acc_mean = mean(acc),
         ies = rt/acc_mean) %>% 
  filter(acc == 1) %>% 
  group_by(subject, freq_consec_value) %>% 
  mutate(outlier = abs(rt-median(rt, na.rm=T))/mad(rt, na.rm=T)) %>%
  filter(outlier <= 4) %>%
  ggplot(aes(freq_consec_lab, ies, group=1)) +
  stat_summary(geom='pointrange') +
  stat_smooth(method='lm', se=FALSE) +
  scale_y_continuous(name='IES (RTs/accuracy)') +
  scale_x_discrete(name='Number of consecutive LOW or HIGH\n frequency preceding items') +
  theme_classic()
ggsave(here('figures/reder2002_ies_freqconsecvalue.tiff'),width=3.5, height=3.5, units='in', compression='lzw')

#############################################################################
# ADDITIONAL PLOTS
#############################################################################
# accuracy by freq and freq_prioritem
test %>% 
  ggplot(aes(freq, acc, fill=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the current item') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='Proportion recalled') +
  coord_cartesian(ylim=c(0.5,1)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1))

# rts by freq and freq_prioritem
test %>%
  group_by(subject) %>% 
  mutate(outlier = abs(rt-median(rt, na.rm=T))/mad(rt, na.rm=T)) %>% 
  filter(acc == 1, outlier <= 4) %>% 
  ggplot(aes(freq, rt, fill=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the recalled item') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='RTs') +
  coord_cartesian(ylim=c(1200,1700)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1))

# rts by freq_consec
test %>%
  group_by(subject) %>% 
  mutate(outlier = abs(rt-median(rt, na.rm=T))/mad(rt, na.rm=T)) %>% 
  filter(acc == 1, outlier <= 4) %>%
  ggplot(aes(freq_consec_lab, rt, group=1)) +
  stat_summary(geom='pointrange') +
  stat_summary(geom='line') +
  theme_classic()

# acc by freqconsec
test %>%
  filter(abs(freq_consec_value) <= 2) %>% 
  group_by(subject) %>% 
  ggplot(aes(freq_consec_lab, acc, group=1)) +
  stat_summary(geom='pointrange') +
  stat_summary(geom='line') +
  scale_x_discrete(name='Number of consecutive LOW or HIGH frequency\npreceding items') +
  theme_classic()



#############################################################################
# ANALYSES
#############################################################################
# calculate IES
model_dat <- test %>%
  group_by(subject, freq, freq_consec_value) %>% 
  mutate(acc_mean = mean(acc),
         ies = rt/acc_mean) %>% 
  filter(acc == 1) %>% 
  group_by(subject, freq_consec_value) %>% 
  mutate(outlier = abs(rt-median(rt, na.rm=T))/mad(rt, na.rm=T)) %>%
  filter(outlier <= 4)

# mixed effects logistic regression of acc ~ freq_prioritem
ml0 <- lmer(ies ~  freq + (1|subject) + (1|stim), data=model_dat)
ml1 <- lmer(ies ~  freq + freq_prioritem + (1|subject) + (1|stim), data=model_dat)
ml2 <- lmer(ies ~  freq*freq_prioritem + (1|subject) + (1|stim), data=model_dat)
sink(here('output/reder2002_lmer1.txt'))
summary(ml1)
cat('\n')
anova(ml0, ml1, ml2)
sink()


# mixed effects logistic regression of acc ~ freq_prioritem
ml3.0 <- lmer(ies ~  freq + (1|subject) + (1|stim), data=model_dat)
ml3 <- lmer(ies ~  freq + freq_consec_value + (1|subject) + (1|stim), data=model_dat)
ml4 <- lmer(ies ~  freq*freq_consec_value + (1|subject) + (1|stim), data=model_dat)
sink(here('output/reder2002_lmer2.txt'))
summary(ml3)
cat('\n')
anova(ml3.0, ml3, ml4)
sink()