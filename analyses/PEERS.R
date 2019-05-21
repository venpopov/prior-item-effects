####
# The data reported here are from the Penn Electrophysiology of Encoding and Retrieval dat (PEERS)
# Experiment 1. 7 sessions, each session contains 16 lists of 16 words presented one at a time
# each dat list was followed by a free recall test
# check this paper for description http://memory.psych.upenn.edu/files/pubs/HealEtal14.pdf

# E1 - sessions 0 to 6
# E2 - sessions 7 to 12
# E3 - session 13 to 19

# From http://memory.psych.upenn.edu/files/pubs/KuhnEtal18.pdf :
# We analyze final free-recall data collected as part of the Penn
# Electrophysiology of Encoding and Retrieval Study (PEERS).
# Subjects recruited to PEERS took part in three subsequently administered multisession experiments, comprising a total of 20
# experimental sessions (of the 171 participants who completed the
#                        seven sessions of Experiment 1, 158 also completed six sessions of
#                        Experiment 2, and 151 completed the remaining sessions of Experiment 3). Subjects in these experiments studied and then freely
# recalled lists of 16 common words under various conditions (immediate free recall in Experiments 1 and 3; immediate, delayed,
#                                                             and continual-distractor free recall [CDFR] in Experiment 2).
# During half of the experimental sessions, subjects were also given
# a final-free-recall test, and the present paper is the first to report on
# these data. Experiment 3 differed from Experiment 1 in that a
# subset of subjects were asked to verbalize all words that came to
# mind during recall


#############################################################################
# SETUP
#############################################################################

rm(list=ls())
library(tidyverse)
library(here)
library(lme4)
library(cowplot)
source(here('analyses/prior_item_functions.R'))

mean_se2 <- function (x) {
  x <- stats::na.omit(x)
  se <- 1.96 * sqrt(stats::var(x)/length(x))
  mean <- mean(x)
  data.frame(y = mean, ymin = mean - se, ymax = mean + se)
}

theme_set(theme_classic(base_size=8))


#############################################################################
# DATA
#############################################################################

#save data as csv
dat <- read.csv('data/PEERS.csv')


dat <- dat %>% 
  group_by(subject, session, listid) %>% 
  arrange(subject, session, listid, test_position) %>% 
  mutate(resplag = study_position-c(NA,study_position[-length(study_position)])) %>% 
  mutate(resplag = ifelse(!is.na(test_position), resplag, NA))

dat <- dat %>% 
  mutate(exp = case_when(
    session <= 6 ~ 1,
    session > 6 & session <= 12 ~ 2,
    TRUE ~ 3
  ))

dat$sp_cat <- ifelse(dat$study_position <= 8, 'first half','second half')



#############################################################################
# PLOTS
#############################################################################
dat1 <- dat %>% 
  filter(exp == 1)


# main effect of prior freq
(f1 <- dat1 %>% 
  ggplot(aes(ave_freq_prioritem1, acc)) +
  stat_summary(geom='point') +
  geom_smooth(method='lm', se=F) +
  scale_x_continuous('Mean log(freq) of preceding item during study', breaks=c(1,1.5,2,2.5,3,3.5,4)) +
  scale_y_continuous('P(recall)'))

(f2 <- dat %>% 
  filter(exp == 1) %>% 
  ggplot(aes(ave_freq_prioritem1, rt)) +
  stat_summary(geom='point') +
  geom_smooth(method='lm', se=F) +
  scale_x_continuous('', breaks=c(1,1.5,2,2.5,3,3.5,4)) +
  scale_y_continuous('Response times (in ms.)'))

plot_grid(f1, f2)
ggsave('figures/peers_freqprioritem1.tiff', units='in', width=7, height=3.5, compression='lzw')

# interaction with lag
(f3 <- dat1 %>% 
  filter(exp == 1) %>% 
  gather(priorlag, priorfreq, ave_freq_prioritem1:ave_freq_prioritem3, ave_freq_prioritem4) %>% 
  filter(!is.na(priorfreq)) %>% 
  mutate(mean_acc = mean(acc, na.rm=T)) %>%
  group_by(priorlag) %>%
  mutate(acc = acc-mean(acc, na.rm=T)+mean_acc) %>%
  ggplot(aes(priorfreq, acc, color=as.factor(priorlag))) +
  stat_summary(geom='point') +
  geom_smooth(method='lm', se=F) +
  scale_x_continuous('Mean log(freq) of preceding item during study', breaks=c(1,1.5,2,2.5,3,3.5,4)) +
  scale_y_continuous('P(recall)') +
  scale_color_discrete('Prior item lag', labels=c(1,2,3,4)))
ggsave('figures/PEERS_priorfreq_lag.tiff', f3+theme(legend.position='bottom'), units='in', width=5, height=5, compression='lzw')



#############################################################################
# models
#############################################################################
dat$freq_cat <- ifelse(dat$Lg10WF <= median(dat$Lg10WF), 'LF','HF')
ml1 <- lmer(acc ~ Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
ml2 <- lmer(acc ~ freq_prioritem1 + Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
anova(ml1, ml2)


# rts
rtml1 <- lmer(rt ~ Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
rtml2 <- lmer(rt ~ freq_prioritem1 + Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
anova(rtml1, rtml2)


# lag effects
mldata <- dat %>% 
  filter(exp == 1, !is.na(freq_prioritem4))
ml1 <- lmer(acc ~ Lg10WF + (1|subject) +(1|word), data=mldata, family='binomial', nAGQ=0)
ml2 <- glmer(acc ~ Lg10WF + freq_prioritem1 + (1|subject) +(1|word), data=mldata, family='binomial', nAGQ=0)
ml3 <- glmer(acc ~ Lg10WF + freq_prioritem2 + (1|subject) +(1|word), data=mldata, family='binomial', nAGQ=0)
ml4 <- glmer(acc ~ Lg10WF + freq_prioritem3 + (1|subject) +(1|word), data=mldata, family='binomial', nAGQ=0)
ml5 <- glmer(acc ~ Lg10WF + freq_prioritem4 + (1|subject) +(1|word), data=mldata, family='binomial', nAGQ=0)

summary(ml2)
summary(ml3)
summary(ml4)
summary(ml5)

exp(fixef(ml2)[3])
exp(fixef(ml3)[3])
exp(fixef(ml4)[3])
exp(fixef(ml5)[3])

#############################################################################
# Condition on whether n-1 was recalled
#############################################################################

dat <- dat %>% 
  arrange(subject, session, listid, study_position) %>% 
  group_by(subject, session, listid) %>% 
  prior_item_analysis('acc','acc')

dat <- mutate(dat, acc = as.numeric(acc))

(f1 <- dat %>% 
    filter(!is.na(acc_prioritem),ave_freq_prioritem1 < 4) %>% 
    ggplot(aes(ave_freq_prioritem1, acc)) +
    stat_summary(geom='point') +
    geom_smooth(se=F, method='lm') +
    # scale_x_continuous('Mean log(freq) of preceding item during study', breaks=c(1,1.5,2,2.5,3,3.5,4)) +
    scale_y_continuous('P(recall)')) +
  facet_wrap(~acc_prioritem, scales='free') 


ml2 <- lmer(acc ~ freq_prioritem1 + Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
ml3 <- lmer(acc ~ freq_prioritem1 + Lg10WF + acc_prioritem + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
ml4 <- lmer(acc ~ freq_prioritem1*acc_prioritem + Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)
ml5 <- lmer(acc ~ freq_prioritem1*acc_prioritem + freq_prioritem1*Lg10WF + (1|subject) + (1|word), data=filter(dat, exp == 1, !is.na(freq_prioritem1)), REML=F)

summary(ml4)
anova(ml3, ml4)