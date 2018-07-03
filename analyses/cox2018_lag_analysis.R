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


#############################################################################
# PRIOR ITEM ANALYSIS
#############################################################################
# split into study and test, extract info about prior item freq during study
# then join with test

dat <- read.csv(here('data/cox2018.csv'))
# dat$acc <- ifelse(!is.na(dat$resp.type.rescore) & (dat$resp.type.rescore == "Hit" | dat$resp.type.rescore == 'CR'), 1, as.numeric(dat$resp.type == 'Hit' | dat$resp.type == 'CR'))
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

study <- dat %>%
  filter(procedure == 'study') %>%
  select(subject, block, condition, stim1, stim2, freq1, freq2) %>%
  mutate(freq = (freq1+freq2)/2) %>%
  group_by(subject, block) %>%
  prior_item_analysis('freq','freq',NULL, max_lag = 3) %>% 
  mutate_at(vars(contains('freq')), as.numeric)

test <- dat %>%
  filter(procedure == 'test') %>% 
  mutate(resp.string = toupper(resp.string))

dat <- test %>%
  filter(condition != 'Free recall', condition != 'Lexical decision') %>% 
  ungroup() %>%
  left_join(study %>% select(-stim2, -freq1, -freq2, -condition), by=c('subject','stim1', 'block')) %>%
  left_join(study %>% select(-stim1, -freq1, -freq2, -condition), by=c('subject','stim2', 'block')) %>%
  left_join(study %>% select(-stim1, -freq1, -freq2, -condition), by=c('subject','stim1'='stim2', 'block')) %>% 
  left_join(study %>% select(-stim2, -freq1, -freq2, -condition), by=c('subject','stim2'='stim1', 'block'))

dat$freq_prioritem1 <- dat %>% select(contains('prioritem1')) %>% rowMeans(na.rm=T)
dat$freq_prioritem2 <- dat %>% select(contains('prioritem2')) %>% rowMeans(na.rm=T)
dat$freq_prioritem3 <- dat %>% select(contains('prioritem3')) %>% rowMeans(na.rm=T)
dat$freq_current <- dat %>% select(contains('freq.')) %>% rowMeans(na.rm=T)
# dat$freq_prioritem4 <- dat %>% select(contains('prioritem4')) %>% rowMeans(na.rm=T)
dat <- dat %>% select(-contains('.x'), -contains('.y'))

# do the same for free recall
free <- study %>% 
  filter(condition == 'Free recall') %>% 
  left_join(select(test, subject, block, condition, resp.string, acc, rt), by=c('condition','subject','block','stim1'='resp.string')) %>% 
  left_join(select(test, subject, block, condition, resp.string, acc, rt), by=c('condition','subject','block','stim2'='resp.string'), suffix=c('1','2')) %>% 
  mutate(freq_current = (freq1+freq2)/2) %>% 
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
# PLOTS
#############################################################################



(f1 <- dat %>% 
    left_join(select(sac_fit, subject, condition, stim1, acc_pred)) %>%
    filter(resp.type.rescore %in% c("Hit",'Miss') | condition == 'free recall' | condition == 'single recognition', !is.na(freq_prioritem1), !is.na(freq_prioritem2), !is.na(freq_prioritem3)) %>% 
    gather(lag, freq_prioritems, freq_prioritem1, freq_prioritem2, freq_prioritem3) %>% 
    mutate(freq_prioritem_cat = cut(freq_prioritems, quantile(freq_prioritems, probs=seq(0,1,length.out=30)), include.lowest = TRUE),
           freq_prioritem_cat = as.numeric(freq_prioritem_cat)) %>%
    group_by(freq_prioritem_cat, lag, condition) %>%
    summarise(acc = mean(acc, na.rm=T),
              acc_pred = mean(acc_pred, na.rm=T),
              rt = mean(rt, na.rm=T),
              IES = rt/acc,
              freq_prioritem = mean(freq_prioritems) %>% round(3)) %>%
    group_by(freq_prioritem_cat, lag) %>% 
    summarise(acc = mean(acc, na.rm=T),
             acc_pred = mean(acc_pred, na.rm=T),
             rt = mean(rt, na.rm=T),
             IES = mean(IES, na.rm=T),
             freq_prioritem = mean(freq_prioritem) %>% round(3)) %>%
    ungroup() %>% 
    gather(data, acc, acc, acc_pred) %>% 
    mutate(data = recode(data, acc='Observed', acc_pred='SAC Model fit')) %>% 
    ggplot(aes(log(10^(freq_prioritem)/51), acc, color=lag)) +
    geom_point(size=1) +
    geom_smooth(se=FALSE, method='lm') +
    theme_bw() +
    theme(panel.grid = element_blank(),
          panel.spacing = unit(1,'lines'),
          legend.position = 'bottom') +
    scale_x_continuous('Mean log(freq) of preceding item during study', breaks = seq(1,4,0.5), limits = c(1,4.2)) +
    scale_color_discrete('Prior item lag', labels=c(1,2,3)) +
    ylab('Hit rate') +
   coord_cartesian(ylim=c(0.35,0.46)) +
    facet_wrap(~data))
  

ggsave('figures/cox2018_lag_fit1.tiff', width=5.8, height=3.5, units='in', compression='lzw')

#############################################################################
# ANALYSES
#############################################################################
mldata <- filter(dat, resp.type %in% c("Hit",'Miss') | condition == "free recall", !is.na(freq_prioritem1)) %>% mutate(acc = as.numeric(acc>0))
ml1 <- glmer(acc ~ condition + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
ml2 <- glmer(acc ~ condition + freq_prioritem1 + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
ml3 <- glmer(acc ~ condition + freq_prioritem2 + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)
ml4 <- glmer(acc ~ condition + freq_prioritem3 + (1|subject) +(1|stim1), data=mldata, family='binomial', nAGQ=0)

