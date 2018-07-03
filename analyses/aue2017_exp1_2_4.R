# Reanalyzes data from the following experiment: 
#   Aue, W. R., Criss, A. H., & Novak, M. D. (2017). Evaluating mechanisms of 
#   proactive facilitation in cued recall. Journal of Memory and Language, 94,
#   103-118.
# Link to paper: https://www.sciencedirect.com/science/article/pii/S0749596X16301565
# LINK to OSF: https://osf.io/6ph2z/
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


#############################################################################
# DATA
#############################################################################
# load and merge experiment 1,2 and 4 (experiment 3 tested list 1, so not relevant)

exp1 <- read.csv(here('data/aue-2017/Exp01_OSF.csv'))
exp2 <- read.csv(here('data/aue-2017/Exp02_OSF.csv'))
exp4 <- read.csv(here('data/aue-2017/Exp04_OSF.csv'))
exp4 <- exp4 %>% 
  mutate(Expt = 4) %>% 
  rename('StudyTrial' = 'Study2Trial')
dat <- bind_rows(exp1, exp2, exp4)
names(dat) <- tolower(names(dat))
dat <- dat %>% 
  rename('subject' = 'ssn') %>% 
  mutate(cond = ifelse(cond == 1, 'rep', 'new')) %>% 
  arrange(expt, subject, studytrial) %>% 
  group_by(expt, subject, expt) %>% 
  prior_item_analysis('cond','cond','new', max_lag = 3) %>% 
  arrange(testtrial) %>% 
  filter(!is.na(cond_prioritem))

# load sac fit
sac_fit_raw <- read.csv('output/aue2017_sac_model_fit.csv')
sac_fit_smry <- read.csv('output/aue2017_smry_sac_model_fit.csv')

#############################################################################
# MAIN PLOTS
#############################################################################

(f1 <- dat %>% 
  group_by(subject, expt, cond, cond_prioritem1) %>% 
  summarise(ncorr = mean(ncorr)) %>% 
  ggplot(aes(cond, ncorr, fill=cond_prioritem1, group=cond_prioritem1)) + 
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  geom_point(data=sac_fit_smry, aes(y=ncorr_pred), position=position_dodge(0.3)) +
  scale_x_discrete(name='Cuetype of the recalled item') +
  scale_fill_discrete(name='Cuetype of the preceeding\nitem during encoding') +
  scale_y_continuous(name='Recall proportion') +
  coord_cartesian(ylim=c(0,0.5)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)))
ggsave(here('figures/aue2017_acc_cuetype_cuetypeprioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')


(f2 <- dat %>% 
  group_by(subject, expt, cond, cond_prioritem) %>%
  summarise(conf = mean(conf)) %>% 
  ggplot(aes(cond, conf, fill=cond_prioritem, group=cond_prioritem)) + 
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Cuetype of the recalled item') +
  scale_fill_discrete(name='Cuetype of the preceeding\nitem during encoding') +
  scale_y_continuous(name='Confidence') +
  coord_cartesian(ylim=c(2.2,4)) +
  theme_classic() +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)))
ggsave(here('figures/aue2017_conf_cuetype_cuetypeprioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')

plot_grid(f1 + theme(legend.position = 'none'), f2)
ggsave(here('figures/aue2017_all.tiff'), width=6, height=3.5, units='in', compression='lzw')

#############################################################################
# ANALYSES
#############################################################################

# mixed effects logistic regression of recall
ml0 <- glmer(ncorr ~  cond + (1|subject:expt) + (1|target), data=dat, family='binomial')
ml1 <- glmer(ncorr ~  cond + cond_prioritem + (1|subject:expt) + (1|target), data=dat, family='binomial')
ml2 <- glmer(ncorr ~  cond*cond_prioritem + (1|subject:expt) + (1|target), data=dat, family='binomial')
sink(here('output/aue2017_glmer1.txt'))
summary(ml2)
cat('\n')
anova(ml0, ml1, ml2)
sink()

# mixed effects logistic regression of confidence
ml3.0 <- lmer(conf ~  cond + (1|subject:expt) + (1|target), data=dat)
ml3 <- lmer(conf ~  cond + cond_prioritem + (1|subject:expt) + (1|target), data=dat)
ml4 <- lmer(conf ~  cond*cond_prioritem + (1|subject:expt) + (1|target), data=dat)
sink(here('output/aue2017_lmer2_conf.txt'))
summary(ml4)
cat('\n')
anova(ml3.0, ml3, ml4)
sink()

