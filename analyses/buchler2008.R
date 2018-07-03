# Reanalyzes data from the following experiment: 
#   Buchler, N. G., Light, L. L., & Reder, L. M. (2008). Memory for items and associations: 
#   Distinct representations and processes in associative recognition. 
#   Journal of Memory and Language, 59(2), 183–199.
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
theme_set(theme_classic(base_size = 8))

#############################################################################
# DATA
#############################################################################

# load data
dat <- read.csv(here('data/buchler2008_exp1.csv'))
dat <- mutate(dat, 
              acc = as.numeric(acc_resp == 1),
              word1 = tolower(word1),
              word2 = tolower(word2))

study <- dat %>% 
  filter(procedure == 'study') %>% 
  select(subject, word1, word2) %>% 
  group_by(subject, word1, word2) %>% 
  mutate(rep = 1:length(word1)) %>% 
  group_by(subject) %>% 
  prior_item_analysis('rep','rep', low='1',max_consec = 2, max_lag = 3) %>% 
  mutate(rep_prioritem = as.numeric(rep_prioritem)) %>% 
  ungroup() %>% 
  select(-rep)

study1 <- dat %>% 
  filter(procedure == 'study') %>% 
  select(subject, word1, word2) %>% 
  group_by(subject, word1, word2) %>% 
  mutate(rep = 1:length(word1)) %>% 
  mutate(rep = ifelse(rep == 1, 'weak','strong')) %>% 
  group_by(subject) %>% 
  prior_item_analysis('rep','rep', low='weak',max_consec = 2, max_lag = 3) %>% 
  mutate(rep_prioritem = as.numeric(rep_prioritem)) %>% 
  ungroup() %>% 
  select(-rep)

test <- dat %>% 
  filter(procedure == 'test') %>% 
  left_join(study, by=c('subject','word1')) %>% 
  filter(!is.na(rep_prioritem)) %>% 
  mutate(studied = ifelse(word2.x == word2.y,'Studied (Hits)','Recombined (FAs)')) %>% 
  filter(condition %in% c('match1and1','rep5and5','swap1and1','swap5and5')) %>% 
  mutate(studied1 = ifelse(grepl('swap', condition), 'Recombined (FAs)','Studied (Hits)')) %>% 
  filter(studied == studied1)

test1 <- dat %>% 
  filter(procedure == 'test') %>% 
  left_join(study1, by=c('subject','word1')) %>% 
  filter(!is.na(rep_prioritem1)) %>% 
  mutate(studied = ifelse(word2.x == word2.y,'Studied (Hits)','Recombined (FAs)')) %>% 
  filter(condition %in% c('match1and1','rep5and5','swap1and1','swap5and5')) %>% 
  mutate(studied1 = ifelse(grepl('swap', condition), 'Recombined (FAs)','Studied (Hits)')) %>% 
  filter(studied == studied1)

test$rep_consec_lab <- test1$rep_consec_lab
test$rep_consec_value <- test1$rep_consec_value

test <- test %>% 
  group_by(subject, word1) %>% 
  filter(rep_prioritem1 == max(rep_prioritem1),
         rep_prioritem2 == max(rep_prioritem2),
         rep_prioritem3 == max(rep_prioritem3)) %>% 
  unique()

smry_for_fit = test %>% 
  filter(studied == 'Studied (Hits)') %>% 
  mutate(rep = ifelse(grepl('5',condition),'strong', 'weak') %>% as.factor()) %>% 
  group_by(rep, subject, rep_prioritem1) %>% 
  summarise(acc = mean(acc, na.rm=T)) %>%
  ungroup() %>% 
  complete(rep, subject, rep_prioritem1) %>% 
  group_by(subject, rep) %>% 
  mutate(acc = ifelse(!is.na(acc), acc, mean(acc, na.rm=T))) %>% 
  group_by(rep, rep_prioritem1) %>% 
  summarise(acc = mean(acc)) %>% 
  mutate(procedure = 'test', studied = 'studied (hits)')

# write.csv(smry_for_fit, 'data/modelling/buchler2008_smry.csv')

#load model fits
sac_fit <- read.csv('output/buchler2008_sac_model_fit.csv') 
sac_fit <- select(sac_fit, subject, word1, word2, acc_pred)
test <- test %>% left_join(sac_fit)

#############################################################################
# PLOTS
#############################################################################
(f1 <-  test  %>% 
  filter(studied == 'Studied (Hits)') %>% 
  mutate(rep = ifelse(grepl('5',condition),'Strong items\n(multiple rep)', 'Weak items\n(one rep)') %>% as.factor()) %>% 
  mutate(rep_prioritem = ifelse(rep_prioritem == 1, 'One','Multiple')) %>% 
  group_by(subject, rep, rep_prioritem) %>% 
  summarise(acc = mean(acc, na.rm=T),
            acc_pred = mean(acc_pred, na.rm=T)) %>% 
  ungroup() %>% 
  Rmisc::normDataWithin(idvar='subject', measurevar = 'acc') %>%
  ggplot(aes(rep, accNormed, fill=rep_prioritem %>% as.factor(), group=rep_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  stat_summary(aes(y=acc_pred), geom="point", position=position_dodge(0.3)) +
  scale_x_discrete(name='Type of pair') +
  scale_fill_discrete(name='Number of repetitions of the\npreceeding item during encoding') +
  scale_y_continuous(name='Hit rate') +
  theme(legend.position = 'bottom'))

ggsave('figures/buchler2008_maineffect.tiff', units='in', width=4, height=5, compression='lzw')


(f2 <- test  %>% 
  filter(studied == "Studied (Hits)", !is.na(rep_prioritem1), !is.na(rep_prioritem2), !is.na(rep_prioritem3)) %>% 
  mutate(rep = ifelse(grepl('5',condition),'Strong items', 'Weak items') %>% as.factor()) %>% 
  filter(rep == 'Weak items') %>% 
  gather(lag, rep_prioritems, rep_prioritem1, rep_prioritem2, rep_prioritem3) %>% 
  mutate(rep_prioritems = ifelse(rep_prioritems == 1, 'One','Multiple'),
         group = paste0(subject, lag)) %>%
  group_by(subject, rep_prioritems, lag, group) %>%
  summarise(acc = mean(acc, na.rm=T),
            acc_pred = mean(acc_pred, na.rm=T)) %>%
  Rmisc::normDataWithin(idvar='group', measurevar = 'acc') %>% 
  Rmisc::normDataWithin(idvar='group', measurevar = 'acc_pred') %>%     
  ggplot(aes(lag, accNormed, fill=rep_prioritems %>% as.factor(), group=rep_prioritems)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  stat_summary(aes(y=acc_predNormed), geom="point", position=position_dodge(0.3)) +
  scale_x_discrete(name='Prior item lag', labels=c('1\n',2,3)) +
  scale_fill_discrete(name='Number of repetitions of the\npreceeding item during encoding') +
  scale_y_continuous(name='Hit rate') +
  coord_cartesian(ylim=c(0.25,0.7)) +
  theme(legend.position = 'bottom'))

ggsave('figures/buchler2008_laginteraction.tiff', units='in', width=4, height=5, compression='lzw')

(f3 <- test %>% 
    filter(studied == 'Studied (Hits)', !grepl('5',condition)) %>% 
    group_by(rep_consec_value, subject) %>% 
    summarise(acc = mean(acc, na.rm=T),
              acc_pred = mean(acc_pred, na.rm=T)) %>%
    ungroup() %>% 
    Rmisc::normDataWithin(idvar='subject', measurevar = 'acc') %>%
    gather(data, acc1, accNormed, acc_pred) %>% 
    mutate(data = recode(data, accNormed='Observed', acc_pred='SAC model fit')) %>% 
    ggplot(aes(rep_consec_value, acc1, fill=data, shape=data, linetype=data)) +
    stat_summary(fun.data = mean_se, geom="line") +
    stat_summary(data=function(x) filter(x, data=='Observed'), geom="linerange") +
    stat_summary(fun.data = mean_se, geom="point", size=2) +
    scale_x_continuous(name='Number of consecutive weak\nor strong preceding items', breaks = c(-1.5,-0.5,0.5,1.5), labels=c('2weak','1weak','1strong','2strong')) +
    scale_shape_manual('', values=c(21,24)) +
    scale_fill_manual('', values=c('black','white')) +
    scale_linetype_discrete('') +
    ylab('Hit rate') +
    theme(legend.position = 'bottom'))

ggsave('figures/buchler2008_consec.tiff', units='in', width=4, height=5, compression='lzw')


(f4 <- test %>% 
  filter(studied == 'Studied (Hits)', !grepl('5',condition)) %>% 
  group_by(subject, rep_prioritem) %>% 
  summarise(acc = mean(acc, na.rm=T),
            acc_pred = mean(acc_pred, na.rm=T)) %>%
  ungroup() %>% 
  complete( subject, rep_prioritem) %>% 
  group_by(subject) %>% 
  mutate(acc = ifelse(!is.na(acc), acc, mean(acc, na.rm=T))) %>% 
  ungroup() %>% 
  mutate(acc_pred = ifelse(!is.na(acc_pred), acc_pred, mean(acc_pred, na.rm=T))) %>%
  gather(data, acc, acc, acc_pred) %>% 
  mutate(data = recode(data, acc='Observed', acc_pred='SAC model fit')) %>% 
  Rmisc::normDataWithin(idvar='subject', measurevar = 'acc') %>%
  ggplot(aes(rep_prioritem, accNormed, group=data, fill=data, shape=data, linetype=data)) +
  stat_summary(data=function(x) filter(x, data=='Observed'), geom='linerange') +
  stat_summary(geom='line') +
  stat_summary(geom='point', size=2) +
  scale_x_continuous(name='Number of repetitions of the\npreceeding item during encoding') +
  scale_y_continuous(name='Hit rate') +
  scale_shape_manual('', values=c(21,24)) +
  scale_fill_manual('', values=c('black','white')) +
  scale_linetype_discrete('') +
  theme(legend.position = 'bottom'))

legend1 <- g_legend(f1 + theme(legend.position = 'bottom'))
plot_grid(f1 + coord_cartesian(ylim=c(0, 1)) + theme(legend.position='none'),
          f2 + coord_cartesian(ylim=c(0, 1)) + scale_y_continuous('Hit rate', breaks=c(0, 0.25,0.5,0.75,1)) + theme(legend.position=c(1,1), legend.justification = c(1,1)))
ggsave('figures/buchler2008_f1f2_fits.tiff', units='in', width=5.5, height=2.8, compression='lzw')

plot_grid(f4 + theme(legend.position='none'),
          f3 + theme(legend.position=c(1,0.05), legend.justification = c(1,0.05)))
ggsave('figures/buchler2008_f3f4_fits.tiff', units='in', width=5.5, height=2.8, compression='lzw')

  

# -------------------------------------------------------------------
# ANALYSES
# -------------------------------------------------------------------

# main analyses
lmdat <- test  %>% 
  filter(studied =="Studied (Hits)") %>% 
  mutate(rep = ifelse(grepl('5',condition),'Strong items\n(multiple rep)', 'Weak items\n(one rep)') %>% as.factor()) %>% 
  mutate(rep_prioritem1 = ifelse(rep_prioritem1 == 1, 'One','Multiple')) %>% 
  mutate(rep_prioritem2 = ifelse(rep_prioritem2 == 1, 'One','Multiple')) %>% 
  mutate(rep_prioritem3 = ifelse(rep_prioritem3 == 1, 'One','Multiple'))

ml0 <- glmer(acc ~ rep + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat)
ml1 <- glmer(acc ~ rep + rep_prioritem1 + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat)
ml2 <- glmer(acc ~ rep*rep_prioritem1 + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat)
sink(here('output/buchler2008.txt'))
summary(ml2)
cat('\n')
anova(ml0, ml1, ml2)
sink()


# interaction with lag
ml3.0 <- glmer(acc ~ rep + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat %>% filter(!is.na(rep_prioritem2)), control=glmerControl(optimizer='bobyqa'))
ml3.1 <- glmer(acc ~ rep + rep_prioritem2 + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat %>% filter(!is.na(rep_prioritem2)), control=glmerControl(optimizer='bobyqa'))
ml3.2 <- glmer(acc ~ rep*rep_prioritem2 + (1|subject) + (1|word1:word2.x), family='binomial', 
               data=lmdat %>% filter(!is.na(rep_prioritem2)), 
               control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun=1e6)))
ml4.0 <- glmer(acc ~ rep + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat %>% filter(!is.na(rep_prioritem3)), control=glmerControl(optimizer='bobyqa'))
ml4.1 <- glmer(acc ~ rep + rep_prioritem3 + (1|subject) + (1|word1:word2.x), family='binomial', data=lmdat %>% filter(!is.na(rep_prioritem3)), control=glmerControl(optimizer='bobyqa'))
ml4.2 <- glmer(acc ~ rep*rep_prioritem3 + (1|subject) + (1|word1) + (1|word2.x), family='binomial', data=lmdat %>% filter(!is.na(rep_prioritem3)), control=glmerControl(optimizer='bobyqa'))
# sink(here('output/buchler2008.txt'))
# cat('\n')

anova(ml3.0, ml3.1, ml3.2)
anova(ml4.0, ml4.1, ml4.2)
# sink()
  

ml5 <- glmer(acc ~  rep+ rep:rep_prioritem1 + rep:rep_prioritem2 + rep:rep_prioritem3 + (1|subject) + (1|word1) + (1|word2.x), 
             family='binomial', control=glmerControl(optimizer='bobyqa'), 
             data=lmdat %>% filter(!is.na(rep_prioritem1),!is.na(rep_prioritem2),!is.na(rep_prioritem3)))




