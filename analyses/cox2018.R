# Reanalyzes data from the following experiment: 
#   Cox, G. E., Hemmer, P., Aue, W. R., & Criss, A. H. (2018). Information and processes 
#   underlying semantic and episodic memory across tasks, items, and individuals. 
#   Journal of Experimental Psychology: General, 147(4), 545.
# Link to paper: https://www.ncbi.nlm.nih.gov/pubmed/29698028
# LINK to OSF: https://osf.io/dd8kp/ 
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
theme_set(theme_bw(base_size=8))


#############################################################################
# PRIOR ITEM ANALYSIS
#############################################################################
# split into study and test, extract info about prior item freq during study
# then join with test

# The original dataset is available from https://osf.io/dd8kp/. The csv file below
# contains the same data with light preprocessing.

dat <- read.csv(here('data/cox2018.csv'))
dat$acc <- ifelse(!is.na(dat$resp.type.rescore) & (dat$resp.type.rescore == "Hit" | dat$resp.type.rescore == 'CR'), 
                  1, 
                  as.numeric(dat$resp.type == 'Hit' | dat$resp.type == 'CR'))
# Get SUBTLEX WORD FREQUENCY
freq <- read.csv(here('materials/SUBTLEX_all_words.csv'))   
freq <- freq %>% select(Word, Lg10WF) %>% mutate(Word = toupper(Word))
dat <- dat %>% 
  select(-freq1, -freq2) %>% 
  left_join(freq, by=c('stim1'='Word')) %>% 
  left_join(freq, by=c('stim2'='Word'), suffix=c('1','2')) %>% 
  rename('freq1'='Lg10WF1', 'freq2'='Lg10WF2')

# remove unfinished blocks
dat <- dat %>% 
  group_by(subject, procedure, block) %>% 
  mutate(count = length(block)) %>% 
  filter(!(count != 20 & condition != 'Free recall')) %>% 
  ungroup() 

# extract the average frequency of the preceding study item
study <- dat %>%
  filter(procedure == 'study') %>%
  select(subject, block, condition, stim1, stim2, freq1, freq2) %>%
  mutate(freq = (freq1+freq2)/2) %>%
  group_by(subject, block) %>%
  prior_item_analysis('freq','freq',NULL, max_lag = 1) %>% 
  mutate_at(vars(contains('freq')), as.numeric) %>% 
  select(-freq_prioritem1)
  # mutate(freq_prioritem = (freq_prioritem1+freq_prioritem2+freq_prioritem3)/3) %>% 
  # select(-freq_prioritem1,-freq_prioritem2,-freq_prioritem3)

test <- dat %>%
  filter(procedure == 'test') %>% 
  mutate(resp.string = toupper(resp.string))

dat <- test %>%
  filter(condition != 'Free recall', condition != 'Lexical decision') %>% 
  ungroup() %>%
  left_join(study %>% select(-stim2, -freq1, -freq2, -freq, -condition), by=c('subject','stim1', 'block')) %>%
  left_join(study %>% select(-stim1, -freq1, -freq2, -freq, -condition), by=c('subject','stim2', 'block')) %>%
  left_join(study %>% select(-stim1, -freq1, -freq2, -freq, -condition), by=c('subject','stim1'='stim2', 'block')) %>% 
  left_join(study %>% select(-stim2, -freq1, -freq2, -freq, -condition), by=c('subject','stim2'='stim1', 'block'))

dat$freq_prioritem <- dat %>% select(contains('prioritem')) %>% rowMeans(na.rm=T)
# dat <- dat %>% select(-freq_prioritem.x, -freq_prioritem.y, -freq_prioritem.x.x, -freq_prioritem.y.y)

# do the same for free recall
free <- study %>% 
  filter(condition == 'Free recall') %>% 
  left_join(select(test, subject, block, condition, resp.string, acc, rt), by=c('condition','subject','block','stim1'='resp.string')) %>% 
  left_join(select(test, subject, block, condition, resp.string, acc, rt), by=c('condition','subject','block','stim2'='resp.string'), suffix=c('1','2')) %>% 
  mutate(acc1 = ifelse(is.na(acc1),0,1),
         acc2 = ifelse(is.na(acc2),0,1),
         acc = acc1+acc2,
         rt = ifelse(is.na(rt1), rt2, rt1)) %>% 
  select(-acc1,-acc2,-freq, -rt1, -rt2)

#combine free recall witht he others
dat <- bind_rows(dat, free) %>% 
  arrange(subject, block)

dat$condition <- dat$condition %>% tolower()
dat$stim1 <- dat$stim1 %>% tolower()

sac_fit <- read.csv('output/cox2018_sac_model_fit.csv')
# fix free recall accuracy, because here it's coded as 2 for recalling the pair
sac_fit[sac_fit$condition == 'free recall',]$acc_pred <- sac_fit[sac_fit$condition == 'free recall',]$acc_pred*2

#############################################################################
# PLOTS DATA
#############################################################################

(f1 <- dat %>% 
  # filter(resp.type %in% c("Hit",'Miss') | condition == "Free recall", !is.na(freq_prioritem)) %>% 
  ungroup() %>% 
  mutate(freq_prioritem_cat = cut(freq_prioritem, quantile(freq_prioritem, probs=seq(0,1,length.out=20), na.rm=T), include.lowest = TRUE),
         freq_prioritem_cat = as.numeric(freq_prioritem_cat)) %>%
  filter(resp.type %in% c("Hit",'Miss') | condition == "free recall", !is.na(freq_prioritem)) %>% 
  group_by(freq_prioritem_cat, condition) %>%
  summarise(acc = mean(acc, na.rm=T),
            rt = mean(rt, na.rm=T),
            IES = rt/acc,
            freq_prioritem = mean(freq_prioritem) %>% round(3)) %>%
  mutate(condition = factor(condition, labels = c('b1) Associative recogniution', 'c1) Cued recall', 'd1) Free recall', 'a1) Single recognition')),
         condition = factor(condition, levels = sort(levels(condition)))) %>% 
  ggplot(aes(log(10^(freq_prioritem)/51), acc)) +
  geom_point() +
  geom_smooth(se=FALSE, method='lm') +
  facet_wrap(~condition, scales='free', nrow=1) +
  theme_bw() +
  theme(panel.grid = element_blank()) +
  scale_x_continuous('Mean log freq of preceding item during study', breaks = c(1,2,3,4), limits = c(1,4.2)) +
  ylab('Hit rate / Proportion recalled'))

(f2 <- dat %>% 
  filter(resp.type %in% c("Hit",'Miss') | condition == "free recall", !is.na(freq_prioritem)) %>% 
  mutate(freq_prioritem_cat = cut(freq_prioritem, quantile(freq_prioritem, probs=seq(0,1,length.out=20)), include.lowest = TRUE),
         freq_prioritem_cat = as.numeric(freq_prioritem_cat)) %>%
  group_by(freq_prioritem_cat, condition) %>%
  filter(acc == 1) %>% 
  summarise(acc = mean(acc, na.rm=T),
            rt = mean(rt, na.rm=T),
            IES = rt/acc,
            freq_prioritem = mean(freq_prioritem) %>% round(3)) %>%
  mutate(condition = factor(condition, labels = c('b2) Associative recogniution', 'c2) Cued recall', 'd2) Free recall', 'a2) Single recognition')),
         condition = factor(condition, levels = sort(levels(condition)))) %>%
  ggplot(aes(log(10^(freq_prioritem)/51), rt)) +
  geom_point() +
  geom_smooth(se=FALSE, method='lm') +
  facet_wrap(~condition, scales='free', nrow=1) +
  theme_bw() +
  theme(panel.grid = element_blank()) +
  scale_x_continuous('Mean log freq of preceding item during study', breaks = c(1,2,3,4), limits = c(1,4.2)) +
  ylab('Response times (in s.)'))

(f_all <- plot_grid(f1, f2, nrow=2))
ggsave('figures/cox2018.tiff', f_all, width=8, height=5, units='in', scale=1, compression='lzw')




#############################################################################
# PLOTS WITH FITS
#############################################################################
dat1 <- dat %>% 
  left_join(select(sac_fit, subject, condition, stim1, acc_pred)) %>% 
  ungroup() %>% 
  mutate(freq_prioritem_cat = cut(freq_prioritem, quantile(freq_prioritem, probs=seq(0,1,length.out=20), na.rm=T), include.lowest = TRUE),
         freq_prioritem_cat = as.numeric(freq_prioritem_cat)) %>%
  filter(resp.type.rescore %in% c("Hit",'Miss') | is.na(resp.type.rescore), !is.na(freq_prioritem)) %>% 
  group_by(freq_prioritem_cat, condition) %>%
  summarise(acc = mean(acc, na.rm=T),
            acc_pred = mean(acc_pred, na.rm=T),
            rt = mean(rt, na.rm=T),
            IES = rt/acc,
            freq_prioritem = mean(freq_prioritem) %>% round(3)) %>% 
  gather(data, acc, acc, acc_pred) %>% 
  mutate(data = recode(data, acc='Observed', acc_pred='SAC Model fit')) %>% 
  mutate(condition = factor(condition, labels = c('b1) Associative recogniution', 'c1) Cued recall', 'd1) Free recall', 'a1) Single recognition')),
         condition = factor(condition, levels = sort(levels(condition)))) 


dat1 %>% 
  ggplot(aes(condition, acc, shape=data, linetype=data, group=data)) +
  stat_summary() +
  stat_summary(geom='line') +
  scale_x_discrete(name='Type of memory test', labels=c('SR','AR','CR','FR')) +
  scale_y_continuous('Hit rate') +
  scale_shape_discrete(name='') +
  scale_linetype_discrete(name='') +
  theme_classic() +
  theme(legend.position = 'bottom')
ggsave('figures/cox2018_overall_fit.tiff', width=2.7, height=2.9, units='in', scale=1, compression='lzw')



(f1 <- dat1 %>% 
   ggplot(aes(log(10^(freq_prioritem)/51), acc, color=data, shape=data, linetype=data)) +
   geom_point() +
   geom_smooth(se=FALSE, method='lm') +
   facet_wrap(~condition, scales='free', nrow=1) +
   theme(panel.grid = element_blank()) +
   scale_x_continuous('Mean log(freq) of preceding item during study', breaks = seq(1,4,0.5), limits = c(1,4.2)) +
   ylab('Hit rate') +
   scale_shape_discrete(name='') +
   scale_linetype_discrete(name='') +
   scale_color_discrete(name='') +
   theme(legend.position='bottom'))



(f2 <- dat %>% 
    filter(resp.type %in% c("Hit",'Miss') | is.na(resp.type.rescore), !is.na(freq_prioritem)) %>% 
    mutate(freq_prioritem_cat = cut(freq_prioritem, quantile(freq_prioritem, probs=seq(0,1,length.out=20)), include.lowest = TRUE),
           freq_prioritem_cat = as.numeric(freq_prioritem_cat)) %>%
    group_by(freq_prioritem_cat, condition) %>%
    filter(acc >= 1) %>% 
    summarise(acc = mean(acc, na.rm=T),
              rt = mean(rt, na.rm=T),
              IES = rt/acc,
              freq_prioritem = mean(freq_prioritem) %>% round(3)) %>%
    mutate(condition = factor(condition, labels = c('b2) Associative recogniution', 'c2) Cued recall', 'd2) Free recall', 'a2) Single recognition')),
           condition = factor(condition, levels = sort(levels(condition)))) %>% 
    filter(!(condition == 'a2) Single recognition' & rt > 1.3)) %>% # remove outlier
    ggplot(aes(log(10^(freq_prioritem)/51), rt)) +
    geom_point() +
    geom_smooth(se=FALSE, method='lm') +
    facet_wrap(~condition, scales='free', nrow=1) +
    theme(panel.grid = element_blank()) +
    scale_x_continuous('Mean log(freq) of preceding item during study', breaks = seq(1,4,0.5), limits = c(1,4.2)) +
    scale_y_continuous(labels=function(x) format(x, digits=3, nsmall=1)) +
    ylab('Response times (in s.)'))



(f_all <- plot_grid(f1 + theme(legend.position = 'top'), f2, nrow=2, rel_heights = c(0.57,0.43)))
ggsave('figures/cox2018_fits.tiff', f_all, width=6.5, height=4, units='in', scale=1, compression='lzw')


#############################################################################
# ANALYSES
#############################################################################
mldata <- filter(dat, resp.type %in% c("Hit",'Miss') | condition == "free recall", !is.na(freq_prioritem)) %>% mutate(acc = as.numeric(acc>0))
ml1 <- glmer(acc ~ condition + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
ml2 <- glmer(acc ~ condition + freq_prioritem + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
ml3 <- glmer(acc ~ condition*freq_prioritem + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
anova(ml1,ml2)
anova(ml2,ml3)



