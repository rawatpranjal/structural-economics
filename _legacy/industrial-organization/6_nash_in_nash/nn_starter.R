############################################
# This program is for Coding Exercise 3    #
############################################

rm(list = ls())

#Set working directory
path <- "/Users/pranjal/Desktop/Structural-Economics/industrial-organization/6_nash_in_nash"
setwd(path)

#Read the relevant libraries
source("nn_functions.R")
library(nleqslv)

#Parameterization
xi= 1:3
apar=-1
theta=rep(0.5,3)

#Compute Nash-in-Nash equilibrium
w0=rep(0.5,3)
results <- nleqslv(w0,nashinnash_wrap,xi=xi,apar=apar,theta=theta)
wstar <- results$x

#Equilibrium analysis
nashinnash(w=wstar,xi=xi,apar=apar,theta=theta)






