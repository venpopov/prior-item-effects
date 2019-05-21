# Gregg (1980)

library(tidyverse)

dat <- read.csv("output/final/gregg1980_sim.csv")

dat %>% 
  filter(procedure == 'test') %>% 
  gather(data, acc, acc, acc_pred) %>% 
  ggplot(aes(isi, acc, shape=freq, fill=data, linetype=data)) +
  stat_summary(fun.data = mean_se, geom="point") +
  stat_summary(fun.data = mean_se, geom="line") +
  scale_x_continuous('Inter-stimulus-interval (in s.)', breaks=c(0,10), labels=c(0,10)) +
  scale_y_continuous('Free recall accuracy') +
  scale_shape_manual('', labels=c('High frequency','Low frequency'), values=c(21,24)) +
  scale_fill_manual('',labels=c('Observed','SAC model fit'), values=c('black','white')) +
  scale_linetype_discrete('', labels=c('Observed','SAC model fit')) +
  ggtitle('Gregg, Montgomery and Castaño (1980)') +
  # coord_cartesian(ylim=c(0.3,0.8)) +
  theme_classic() +
  theme(legend.position = c(1,1), legend.justification = c(1,1), legend.spacing = unit(-1, 'lines'), legend.background = element_blank(), title = element_text(size=9))

ggsave('figures/gregg1980.tiff', width=3.5, height=3.5, units='in', compression='lzw')
