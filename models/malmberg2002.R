library(tidyverse)

# -------------------------------------------------------------------
# EXP 2
# -------------------------------------------------------------------

dat <- read.csv('output/final/malmberg2002_exp2_sim.csv')

dat %>% 
  filter(procedure == 'test') %>% 
  gather(data, hits, hits, hits_pred) %>% 
  ggplot(aes(study_duration, hits, shape=freq, fill=data, linetype=data)) +
  stat_summary(fun.data = mean_se, geom="point") +
  stat_summary(fun.data = mean_se, geom="line") +
  scale_x_continuous('Study Time (sec)', breaks=c(0.25,1,3)) +
  scale_y_continuous('Hit rate') +
  scale_shape_manual('', labels=c('High frequency','Low frequency'), values=c(21,24)) +
  scale_fill_manual('',labels=c('Observed','SAC model fit'), values=c('black','white')) +
  scale_linetype_discrete('', labels=c('Observed','SAC model fit')) +
  ggtitle('Malmberg & Nelson (2003), Exp. 2') +
  coord_cartesian(ylim=c(0.3,0.8)) +
  theme_classic() +
  theme(legend.position = c(1,0), legend.justification = c(1,0), legend.spacing = unit(-1, 'lines'), legend.background = element_blank(), title = element_text(size=9))

ggsave('figures/malmberg2003_exp2.tiff', width=3.5, height=3.5, units='in', compression='lzw')


# -------------------------------------------------------------------
# EXP 3
# -------------------------------------------------------------------
dat <- read.csv('output/final/malmberg2002_exp3_sim1.csv')

dat %>% 
  # filter(procedure == 'test') %>% 
  mutate(study_duration = recode(study_duration, `1.45` = 'a) Short study time (1.2 sec)', `4.25` = 'b) Long study time (4.0 sec)')) %>% 
  gather(data, hits, hits, hits_pred) %>% 
  ggplot(aes(partner_freq, hits, shape=target_freq, fill=data, linetype=data, group=interaction(target_freq,data))) +
  stat_summary(fun.data = mean_se, geom="point", size=2) +
  stat_summary(fun.data = mean_se, geom="line") +
  scale_x_discrete('Frequency of word with which the target was studied',labels=c('HF','LF')) +
  scale_y_continuous('Hit rate') +
  scale_shape_manual('', labels=c('HF target','LF target'), values=c(21,24)) +
  scale_fill_manual('',labels=c('Observed','SAC model fit'), values=c('black','white')) +
  scale_linetype_discrete('', labels=c('Observed','SAC model fit')) +
  ggtitle('Malmberg & Nelson (2003), Exp. 3') +
  facet_wrap(~study_duration) +
  coord_cartesian(ylim=c(0.4,0.75)) +
  theme_bw() +
  theme(legend.spacing = unit(-1, 'lines'), legend.background = element_blank(), title = element_text(size=9), panel.grid=element_blank())

ggsave('figures/malmberg2003_exp3.tiff', width=6.5, height=3.1, units='in', compression='lzw')
