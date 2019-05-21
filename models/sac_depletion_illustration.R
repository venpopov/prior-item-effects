##############################################################################
# load simulations of Ward et al (2003) and plot WM depletion for illustration
##############################################################################

library(tidyverse)
library(cowplot)

dat <- read.csv('output/ward2003_results_for_ploting_wm.csv')
study <- dat %>% filter(procedure == 'study')

(f1 <- study %>% 
  filter(subject == 1, list == 'context2') %>% 
  mutate(wmAvailablePost = wmAvailable-wmSpent) %>% 
  gather(wmWhen, wmAvailable, wmAvailable, wmAvailablePost) %>% 
  ggplot(aes(nominal.sp, wmAvailable)) +
  geom_line() +
  geom_point(aes(fill=freq, shape=freq), size=2) +
  ylab('Available resources') +
  xlab('Study position') +
  scale_fill_manual('', labels=c('Strong item','Weak item'), values=c('black','white')) +
  scale_shape_manual('', labels=c('Strong item','Weak item'), values=c(21,24)) +
  theme_classic(base_size=9) +
  coord_cartesian(ylim=c(0,3)) +
  theme(legend.position=c(1,1.1), legend.justification=c(1,1)))


(f2 <- study %>% 
  mutate(wmAvailablePost = wmAvailable-wmSpent) %>% 
  # gather(wmWhen, wmAvailable, wmAvailable, wmAvailablePost) %>% 
  ggplot(aes(nominal.sp, wmAvailable-wmSpent, fill=composition, shape=freq)) +
  stat_summary(geom='line') +
  stat_summary() +
  ylab('Mean available resources') +
  xlab('Study position') +
  scale_fill_manual('', labels=c('Mixed list','Pure list'), values=c('white','black')) +
  scale_shape_manual('', labels=c('Strong item','Weak item'), values=c(21,24)) +
  theme_classic(base_size=9))

plot_grid(f1, f2, nrow=1, rel_widths = c(0.4,0.6))
ggsave('figures/sac_wmdepletion_illustration.tiff', width=6.5, height=2.5, units='in', compression='lzw')


(f1 <- study %>% 
    filter(subject == 1, list == 'context5') %>% 
    mutate(wmAvailablePost = wmAvailable-wmSpent) %>% 
    gather(wmWhen, wmAvailable, wmAvailable, wmAvailablePost) %>% 
    ggplot(aes(nominal.sp, wmAvailable+0.2)) +
    geom_line() +
    geom_point(aes(fill=freq, shape=freq), size=2) +
    ylab('Available resources') +
    xlab('Study position') +
    scale_fill_manual('Instructions', labels=c('TBF','TBR'), values=c('black','white')) +
    scale_shape_manual('Instructions', labels=c('TBF','TBR'), values=c(21,24)) +
    theme_classic(base_size=9) +
    coord_cartesian(ylim=c(0,3)) +
    theme(legend.position=c(1,1.1), legend.justification=c(1,1)))


set.seed(2453)
dat <- data.frame(cuetype = sample(c('TBF','TBR'), 24, replace=T), trial = 1:24)
dat$wmAvailableStart <- NA
dat$wmAvailablePost <- NA
dat$wmSpent <- NA

wmAvailable = 3
delta = 0.639
wmRecovery = 0.71
for (i in 1:nrow(dat)) {
  dat$wmAvailableStart[i] = wmAvailable
  wmRequested1 = 5 * delta ** 2
  wmReceived1 = min(wmRequested1, wmAvailable)
  wmAvailable = wmAvailable-wmReceived1
  wmRequested2 = ifelse(dat$cuetype[i] == 'TBR', 5 * (delta * (1-delta)) ** 2, 0)
  wmReceived2 = min(wmRequested2, wmAvailable)
  wmAvailable = wmAvailable-wmReceived2
  wmSpent = wmReceived1+wmReceived2
  dat$wmAvailablePost[i] <- wmAvailable
  dat$wmSpent[i] <- wmSpent
  wmAvailable <- min(3, wmAvailable + 3 * wmRecovery)
}


dat %>% 
    gather(wmWhen, wmAvailable, wmAvailableStart, wmAvailablePost) %>% 
    ggplot(aes(trial, wmAvailable)) +
    geom_line() +
    geom_point(aes(fill=cuetype, shape=cuetype), size=2) +
    ylab('Available resources') +
    xlab('Study position') +
    scale_fill_manual('Instructions', labels=c('TBF','TBR'), values=c('black','white')) +
    scale_shape_manual('Instructions', labels=c('TBF','TBR'), values=c(21,24)) +
    theme_classic(base_size=9)

ggsave('figures/sac_wmdepletion_illustration_df.tiff', width=6.5, height=2.5, units='in', compression='lzw')
