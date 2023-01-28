############################################
# This program generates numerical results #
# for the Miller-Sheu entry project        #
############################################

rm(list = ls())

#Set working directory
path <- "/Users/pranjal/Desktop/Structural-Economics/industrial-organisation/bertrand-with-logit-demand"

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
