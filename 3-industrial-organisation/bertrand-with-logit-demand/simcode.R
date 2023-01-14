############################################
# This program generates numerical results #
# for the Miller-Sheu entry project        #
############################################

rm(list = ls())

#Set working directory
path <- "/Users/pranjal/Desktop/Structural-Economics/industrial-organisation/bertrand-with-logit-demand" #CHANGED

setwd(path)


#Read the relevant libraries
source("simcode_functions.R")

#############################################################################
# Initial Data Specification
#############################################################################

rawdata <- list("shares"=c(0.15,0.15,0.3,0.3))
rawdata$margin <- 0.5
rawdata$prices <- c(1,1,1,1)
rawdata$fNum <- fNum <- c(1,2,3,4)
rawdata$mktP <- 1
rawdata$mktQ <- 1

print(rawdata)

##########################################################################
# Obtain structural parameters
##########################################################################
calres <- cal.standard(m=rawdata$margin,s=rawdata$shares,
                       p=rawdata$prices,fNum=rawdata$fNum)

#Code check: confirm FOC = 0 with calibrated parameters
f.foc_logit(p=rawdata$prices,c=calres$c,alpha=calres$alpha,
            xsi=calres$xsi,fNum=rawdata$fNum)
print(calres)
  
##########################################################################
# Merger simulation
##########################################################################

#New ownership structure 
fNum2 <- c(1,1,3,4)

#Value of FOC evaluated at pre-merger prices: "Upward Pricing Pressure"
f.foc_logit(p=rawdata$prices,c=calres$c,alpha=calres$alpha,
            xsi=calres$xsi,fNum=fNum2)

#Compute post-merger equilibrium prices
#simres = nleqslv(rawdata$prices,fn=f.foc_logit, #CHANGED
#                 c=calres$c,fNum=fNum2,
#                 alpha=calres$alpha,xsi=calres$xsi)

#simres = nleqslv(rawdata$prices,fn=f.foc_logit, #CHANGED
#                 c=calres$c*c(0.9,0.9,1,1),fNum=fNum2,
#                 alpha=calres$alpha,xsi=calres$xsi)

#simres = nleqslv(rawdata$prices,fn=f.foc_logit, #CHANGED
#                 c=calres$c,fNum=c(1,1,1,1),
#                 alpha=calres$alpha,xsi=calres$xsi)

#simres = nleqslv(rawdata$prices,fn=f.foc_logit, #CHANGED
#                 c=calres$c,fNum=fNum2,
#                 alpha=calres$alpha,xsi=calres$xsi*c(1.1,1.1,1,1))

simres = nleqslv(rawdata$prices,fn=f.foc_logit, #CHANGED
                 c=calres$c*c(1,5,1,1),fNum=fNum2,
                 alpha=calres$alpha,xsi=calres$xsi)

#Storing results
simres$prices <- simres$x
simres$shares <- quant_logit(p=simres$prices,alpha=calres$alpha,xsi=calres$xsi)

#Printing
print(simres)

