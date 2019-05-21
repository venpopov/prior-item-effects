  library(tidyverse)
  
  #############################################################################
  # HULME 1997
  #############################################################################
  
  
  dat <- read.csv('output/final/hulme1997_simulation.csv')
  
  dat %>% 
    filter(procedure == 'test') %>% 
    gather(data, acc, acc, acc_pred) %>% 
    mutate(data = recode(data, acc='Observed', acc_pred='SAC model')) %>% 
    ggplot(aes(sp, acc, fill=data, shape=freq, linetype=data)) +
    stat_summary(fun.data = mean_se, geom="line") +
    stat_summary(fun.data = mean_se, geom="point", size=1) +
    theme_classic(base_size=9) +
    scale_x_continuous('Serial position', breaks=1:7) +
    scale_y_continuous('Proportion correct') +
    scale_shape_manual('', values=c(21,24), labels=c('High frequency','Low frequency')) + 
    scale_linetype('') +
    coord_cartesian(ylim=c(0,1)) +
    scale_fill_manual('', values=c('black','white'))
  
  ggsave('figures/hulme1997.tiff', width=4.5, height=3, units='in', compression='lzw')
  
  
  #############################################################################
  # HULME 2003 - Exp. 1 and Exp. 2
  #############################################################################
  
  
  dat2 <- read.csv('output/final/hulme2003_simulation_exp1_2.csv')
  
  dat2 %>% 
    filter(procedure == 'test') %>% 
    gather(data, acc, acc, acc_pred) %>% 
    mutate(data = recode(data, acc='Observed', acc_pred='SAC model')) %>% 
    ggplot(aes(sp, acc, shape=composition, group=composition, linetype=composition)) +
    stat_summary(fun.data = mean_se, geom="line") +
    stat_summary(fun.data = mean_se, geom="point", size=1) +
    scale_x_continuous('Serial position', breaks=1:7) +
    scale_y_continuous('Proportion correct', breaks=seq(0.1,1,0.1)) +
    scale_shape_discrete('') +
    scale_linetype_manual('', values=c('dashed','dashed','solid','solid')) +
    theme_bw(base_size=9) +
    theme(panel.grid = element_blank(),
          legend.position = c(0.99,1.05),
          legend.justification = c(1,1),
          legend.background = element_blank()) +
    facet_wrap(~data)
  
  ggsave('figures/hulme2003_exp1_2.tiff', width=6.5, height=3.5, units='in', compression='lzw')
  
  
  #############################################################################
  # HULME 2003 - Exp. 3
  #############################################################################
  
  
  dat3 <- read.csv('output/final/hulme2003_simulation_exp3.csv')
  
  dat3 %>% 
    filter(procedure == 'test') %>% 
    gather(data, acc, acc, acc_pred) %>% 
    mutate(data = recode(data, acc='Observed', acc_pred='SAC model')) %>% 
    ggplot(aes(sp, acc, shape=composition, group=composition, linetype=composition)) +
    stat_summary(fun.data = mean_se, geom="line") +
    stat_summary(fun.data = mean_se, geom="point", size=1) +
    scale_x_continuous('Serial position', breaks=1:7) +
    scale_y_continuous('Proportion correct', breaks=seq(0.1,1,0.1)) +
    scale_shape_discrete('') +
    scale_linetype_manual('', values=c('dashed','dashed','solid','solid')) +
    theme_bw(base_size=9) +
    theme(panel.grid = element_blank(),
          legend.position = c(0,0),
          legend.justification = c(0,0),
          legend.background = element_blank()) +
    facet_wrap(~data)
  
  ggsave('figures/hulme2003_exp3.tiff', width=6.5, height=3.5, units='in', compression='lzw')
