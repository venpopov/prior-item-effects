library(tidyverse)
library(cowplot)

base <- function(Tmat, cTime, p=0.8, d=0.18) {
  S = matrix(0, nrow=nrow(Tmat), ncol=ncol(Tmat))
  
  for (t in 1:ncol(Tmat)) {
    Fmat = (1+Tmat[,t]-Tmat)^(-d)
    B = rowSums(S*Fmat, na.rm=T) * ifelse(rowSums(is.na(Fmat)) == ncol(Fmat),NA, 1)
    S[,t] = p*(1-B)
  }
  
  S = round(S,3)
  Fmat = (1+cTime-Tmat)^(-d)
  Fmat[cTime < Tmat] = NaN
  cS = S*Fmat
  cS[cS == Inf] = 0
  B = rowSums(cS, na.rm=T)
  return(B)
}

ind_strengths <- function(Tmat, cTime, p=0.8, d=0.18) {
  S = matrix(0, nrow=nrow(Tmat), ncol=ncol(Tmat))
  
  for (t in 1:ncol(Tmat)) {
    Fmat = (1+Tmat[,t]-Tmat)^(-d)
    B = rowSums(S*Fmat, na.rm=T) * ifelse(rowSums(is.na(Fmat)) == ncol(Fmat),NA, 1)
    S[,t] = p*(1-B)
  }
  
  S = round(S,3)
  Fmat = (1+cTime-Tmat)^(-d)
  Fmat[cTime < Tmat] = NaN
  cS = S*Fmat
  cS[cS == Inf] = NA
  cS[is.na(cS)] = NA
  cS
  return(cS)
}

Tmat <- matrix(c(0,60,120,180,240), nrow=1)


dat <- sapply(seq(0,300,0.01), function(x) base(Tmat, x))
dat = data.frame(time = seq(0, 300, 0.01), base=dat)


str = sapply(seq(0,300,0.1), function(x) ind_strengths(Tmat, x))
str = as.data.frame(t(str))
names(str) = 1:5
str$time = seq(0,300,0.1)
str <- str %>% 
  gather(rep, strength, `1`,`2`,`3`,`4`,`5`) %>% 
  mutate(rep = paste0('Increment #', rep))

f1 <- ggplot(dat, aes(time, base)) +
  geom_line() +
  xlab('Time (s.)') +
  ylab('Base-level strength of node') +
  theme_classic() +
  coord_cartesian(ylim=c(0,1))

f2 <- ggplot(str, aes(time, strength, color=rep)) +
  geom_line(size=0.75) +
  scale_color_discrete('') +
  xlab('Time (s.)') +
  ylab('Strength of increment') +
  theme_classic() +
  theme(legend.position=c(1,1), legend.justification=c(1,1))  +
  coord_cartesian(ylim=c(0,1))

plot_grid(f1,f2, nrow=1)
ggsave('figures/sac_strength_illustration.tiff', width=6.5, height=3, units='in', compression='lzw')
