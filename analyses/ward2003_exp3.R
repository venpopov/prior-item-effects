# Reanalyzes data from the following experiment: 
#   Ward, G., Woodward, G., Stevens, A., & Stinson, C. (2003). Using overt rehearsals 
#   to explain word frequency effects in free recall. Journal of Experimental Psychology: 
#   Learning, Memory, and Cognition, 29(2), 186.
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

mean_se2 <- function (x) {
  x <- stats::na.omit(x)
  se <- 1.96 * sqrt(stats::var(x)/length(x))
  mean <- mean(x)
  data.frame(y = mean, ymin = mean - se, ymax = mean + se)
}

theme_set(theme_classic(base_size=8))


#############################################################################
# PRIOR ITEM ANALYSIS
#############################################################################
# split into study and test, extract info about prior item freq during study
# then join with test

dat <- read.csv(here('data/ward2003_exp3.csv'))
names(dat) <- tolower(names(dat))
names(dat)[names(dat) == 'correct'] <- 'acc'

dat <- dat %>%
  mutate(composition = recode(trialtype, l = 'pure', h='pure',m='mixed')) %>% 
  arrange(subject, list, nominal.sp) %>% 
  group_by(subject, list) %>% 
  prior_item_analysis('freq','freq','LOW',4, 3) %>% 
  filter(!is.na(freq_prioritem))
         # nominal.sp > 2, nominal.sp < 15) # remove buffer items bc of recency/primacy

fit <- read.csv('output/ward2003_sac_model_fit.csv')
fit <- mutate(fit, freq = toupper(freq),
              freq_prioritem = toupper(freq_prioritem))


#############################################################################
# PLOTS
#############################################################################

# accuracy by freq and freq_prioritem
(f1 <- dat %>% 
  filter(trialtype == 'm') %>% 
  ggplot(aes(freq, acc, fill=freq_prioritem, group=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the recalled item') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='b) Proportion recalled') +
  coord_cartesian(ylim=c(0.1,0.6)) +
   ggtitle('') +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)) +
  stat_summary(data=fit, aes(y=acc_pred), geom='point', position=position_dodge(0.3), size=1.5))
ggsave(here('figures/ward2003_exp3_acc_freq_freqprioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')


# number of rehearsals by freq and freq_prioritem
(f2 <- dat %>% 
  filter(trialtype == 'm', nominal.sp > 2, nominal.sp < 15) %>%   
  ggplot(aes(freq, rehearsals, fill=freq_prioritem, group=freq_prioritem)) +
  stat_summary(fun.data = mean_se, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  scale_x_discrete(name='Frequency of the recalled item') +
  scale_fill_discrete(name='Frequency of the preceeding\nitem during encoding') +
  scale_y_continuous(name='c) Number of rehearsals') +
  coord_cartesian(ylim=c(2,3.8)) +
    ggtitle('') +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1)))
ggsave(here('figures/ward2003_exp3_rehearsal_freq_freqprioritem.tiff'), width=3.5, height=3.5, units='in', compression='lzw')

(f3 <- dat %>% 
  filter(composition != '') %>% 
  ggplot(aes(composition, acc, fill=freq)) +
  stat_summary(fun.data = mean_se2, geom="col", position='dodge', width=0.3) +
  stat_summary(fun.data = mean_se2, geom="errorbar", width=0.1, position=position_dodge(0.3)) +
  stat_summary(data=fit, aes(y=acc_pred), geom='point', position=position_dodge(0.3), size=1.5) +
  scale_x_discrete('List composition') +
  scale_y_continuous('a) Proportion recalled') +
  scale_fill_manual('Frequency of the recalled item', values=c('dodgerblue2','darkorange1')) +
  coord_cartesian(ylim=c(0.1,0.75)) +
  ggtitle('Ward et al (2003; Exp. 3)') +
  theme(legend.position = c(1,1),
        legend.justification = c(1,1),
        title = element_text(size=8)))

ggsave(here('figures/ward2003_composition.tiff'), f3, width=2.5, height=2.5, units='in', scale=1, compression='lzw')


# combine plots
legend <- g_legend(f1 + theme(legend.position = 'bottom'))
(f_all <- plot_grid(f3,
                    f1, 
                    f2 + annotate('text', x=1, y=3.5, label='b)'), 
                    nrow = 1))
ggsave(here('figures/ward2003_all_figures.tiff'), f_all, width=7, height=2.8, units='in', scale=1, compression='lzw')

#############################################################################
# ANALYSES
#############################################################################

# mixed effects logistic regression of acc ~ freq_prioritem
dat1 <- filter(dat, nominal.sp > 2, nominal.sp < 15) # remove buffer items recency and primacy
ml0 <- glmer(acc ~  freq + rehearsals + fsp +  (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml1 <- glmer(acc ~  freq + rehearsals + fsp + freq_prioritem + (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml2 <- glmer(acc ~  rehearsals + fsp + freq*freq_prioritem + (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
sink(here('output/ward2003_exp3_lmer1.txt'))
summary(ml2)
cat('\n')
anova(ml0, ml1, ml2)
sink()


# mixed effects logistic regression of acc ~ freq_prioritem
ml0 <- glmer(acc ~  freq + rehearsals + fsp +  (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml1 <- glmer(acc ~  freq + rehearsals + fsp + freq_prioritem + (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
ml2 <- glmer(acc ~  rehearsals + fsp + freq*freq_prioritem + (1|subject) + (1|word), 
             data=dat1, family="binomial", control = glmerControl(optimizer = 'bobyqa'))
sink(here('output/ward2003_exp3_lmer1.txt'))
summary(ml2)
cat('\n')
anova(ml0, ml1, ml2)
sink()