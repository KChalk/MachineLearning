library(ggplot2)
library(tidyverse)
xes <- seq(1,50,1)
y2 <- dpois(xes, lambda =2)
y6 <- dpois(xes, lambda =6)


lambda <- 2
data <- data_frame(x=xes,y2=y2,y6=y6)


l2plot <- ggplot(data,aes(x=x, y=y2))+
  geom_point()


l6plot <- ggplot(data,aes(x=x, y=y6))+
  geom_point()


combplot <-ggplot(data,aes(x=x))+
  geom_point(aes(y=y2),color='blue')+
  geom_point(aes(y=y6),color='purple')+
  geom_line(aes(y=y2),color='blue')+
  geom_line(aes(y=y6),color='purple')

l2plot  
l6plot
combplot

ggsave('l2.png', l2plot)
ggsave('l6.png', l6plot)
ggsave('comp.png', combplot)


xs <- seq(1,10,.1)
g12 <- dgamma(xs,shape=1,rate=2)
g35 <- dgamma(xs,shape=3,rate=5)

data2 <- data_frame(x=xs,g12=g12, g35=g35)

gamplot <-ggplot(data2,aes(x=x))+
  geom_point(aes(y=g12), color='blue')+
  geom_point(aes(y=g35), color='purple')+
  labs(y='Gamma, blue12, purple35')

gamplot
ggsave('gam.png', gamplot)
