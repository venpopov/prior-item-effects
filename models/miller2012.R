library(tidyverse)

dat <- read.csv('output/final/miller2012_simulation.csv')

dat %>% 
  filter(procedure == 'test') %>% 
  gather(data, acc, acc, acc_pred) %>% 
  mutate(data = recode(data, acc='Observed', acc_pred='SAC model')) %>% 
  ggplot(aes(sp, acc, shape=composition, group=composition, linetype=composition, fill=composition)) +
  stat_summary(fun.data = mean_se, geom="line") +
  stat_summary(fun.data = mean_se, geom="point", size=2) +
  scale_x_continuous('Serial position', breaks=1:7) +
  scale_y_continuous('Proportion correct', breaks=seq(0.1,1,0.1)) +
  scale_shape_manual('', values=c(21,24,22,8), labels=c('Mixed HL','Mixed LH','Pure HH','Pure LL')) +
  scale_fill_manual('', values=c('white','white','black','black'), labels=c('Mixed HL','Mixed LH','Pure HH','Pure LL')) +
  scale_linetype_manual('', values=c('dashed','dashed','solid','solid'), labels=c('Mixed HL','Mixed LH','Pure HH','Pure LL')) +
  theme_bw(base_size=9) +
  theme(panel.grid = element_blank(),
        legend.position = c(0.99,1.05),
        legend.justification = c(1,1),
        legend.background = element_blank()) +
  facet_wrap(~data)

ggsave('figures/miller2012.tiff', width=6.5, height=3.5, units='in', compression='lzw')


dat %>% 
  filter(procedure == 'test') %>% 
  gather(data, acc, acc, acc_pred) %>% 
  mutate(data = recode(data, acc='Observed', acc_pred='SAC model')) %>% 
  mutate(composition = recode(composition, `Pure HF` = '1', `Mixed HHHLLL` = '2', `Mixed LLLHHH` = '3', `Pure LF` = '4'),
         composition = as.character(composition)) %>% 
  # filter(sp != 6) %>%
  ggplot(aes(composition, acc, shape=data, linetype=data, fill=data, group=data)) +
  stat_summary(fun.data = mean_se, geom="line") +
  stat_summary(fun.data = mean_se, geom="point", size=2) +
  scale_x_discrete('List composition', labels = c('Pure HH','Mixed HL', 'Mixed LH', 'Pure LL')) +
  scale_y_continuous('Proportion correct', breaks=seq(0.1,1,0.1)) +
  scale_shape_manual('', values=c(21,24)) +
  scale_fill_manual('', values=c('black','white')) +
  scale_linetype_discrete('') +
  theme_bw(base_size=9) +
  theme(panel.grid = element_blank(),
        legend.position = c(0.99,1),
        legend.justification = c(1,1),
        legend.background = element_blank())

ggsave('figures/miller2012_mean.tiff', width=3, height=3, units='in', compression='lzw')
