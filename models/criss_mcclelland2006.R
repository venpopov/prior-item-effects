# Criss & McClelland (2006)

library(tidyverse)

dat <- read.csv("output/final/criss_mcclelland_2006.csv")
  
dat %>% 
  filter(procedure == 'test') %>% 
  gather(data, hits, hits, hits_pred) %>% 
  ggplot(aes(study_duration, hits, shape=freq, fill=data, linetype=data)) +
  stat_summary(fun.data = mean_se, geom="point") +
  stat_summary(fun.data = mean_se, geom="line") +
  scale_x_continuous('Study Time (sec)', breaks=c(0.15,0.30,0.6)) +
  scale_y_continuous('Hit rate') +
  scale_shape_manual('', labels=c('High frequency','Low frequency'), values=c(21,24)) +
  scale_fill_manual('',labels=c('Observed','SAC model fit'), values=c('black','white')) +
  scale_linetype_discrete('', labels=c('Observed','SAC model fit')) +
  ggtitle('Criss & McClelland (2006)') +
  # coord_cartesian(ylim=c(0.3,0.8)) +
  theme_classic() +
  theme(legend.position = c(1,0), legend.justification = c(1,0), legend.spacing = unit(-1, 'lines'), legend.background = element_blank(), title = element_text(size=9))

ggsave('figures/criss_mcclelland_2006.tiff', width=3.5, height=3.5, units='in', compression='lzw')
