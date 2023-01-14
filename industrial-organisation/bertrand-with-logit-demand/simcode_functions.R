#######################################################################
# This library contains functions for the Bertrand logit calibration
# and merger simulation.
#   !!!!! I have intentionally added two bugs !!!!!
######################################################################

# Downloaded libraries
library(nleqslv)


####################################################################
# f.owner converts firm assignment into ownership matrix
#   Also can be used to dummy out observations from different markets
#   Inputs include:
#     vec= vector assigning products to firms (or markes)
#   Output includes:
#     (unnamed) ownership/market matrix
####################################################################

f.owner <- function(vec) {
  C <- matrix(rep(vec,length(vec)),nrow=length(vec))  
  return(C==t(C))  
}



###################################################################
# Function calibrates logit demand from shares, prices, one margin
# Adjusted to account for single or multiproduct firms.
# Imbeds Nash-Bertrand equilibrium concept
#   Inputs include:
#       m = margin of product 1
#       s = vector of shares
#       p = vector of prices
#       fNum = vector assigning products to firms
#   Outputs include:
#       alpha = price coefficient.
#       a = mean non-price valuations
#       c = vector of rationalizing costs
#       type_j = Nocke and Shutz product types, type=exp(a-alpha*c)
#       type = firm-level types
#       dqdp = matrix of demand derivatives
#       div = diversion matrix


cal.standard = function(m,s,p,fNum) {
  
  #Obtaining an ownership matrix
  owner = f.owner(fNum)
  print(owner)
  
  #Number of product
  num_prod <- length(s)
  print(num_prod)
  
  #Obtains cost of product 1 from the margin
  c1 = p[1]*(1-m)
  print(c1)
  
  #Useful temporary matrix
  temp = -tcrossprod(s,s)
  diag(temp) = s*(1-s)
  
  #Obtains price coefficient via FOC of 1st firm
  if (1==length(fNum[fNum==1])){
    #alpha = -1/((1-s[1])*(p[1]-c1))
    alpha = as.vector(-solve(temp[fNum==1,fNum==1])[1,]%*%s[fNum==1]/(p[1]-c1))
  } else {
    alpha = as.vector(-solve(temp[fNum==1,fNum==1])[1,]%*%s[fNum==1]/(p[1]-c1))    
  }
  print(alpha)
  
  #Obtains matrix of demand derivatives
  dqdp = alpha*temp 
  print(dqdp)
  
  #Obtain costs of other products
  c = as.vector(p+solve(owner*t(dqdp))%*%s )
	print(c)
	
  #Obtains mean valuations via logit transformation
  xsi = log(s/(1-sum(s))) - alpha*p
	print(xsi)
	
  #Obtains diversion matrix
  div <- t(matrix(s,num_prod,num_prod))*matrix(1/(1-s),num_prod,num_prod)
  diag(div)<- -1

  #Obtains the Nocke-Schutz (2018) "types" 
  type_j <- exp(xsi-alpha*c)
  type <- aggregate(type_j,by=list(fNum),FUN=sum)$x  

  return(list(alpha=alpha,xsi=xsi,c=c,type=type,type_j=type_j,dqdp=dqdp,div=div))
}


###################################################################
# The following series of functions provide quantities for the four demand systems
#   Inputs include:
#     p = vector of prices
#     alpha = price coefficient (logit)
#     xsi = vector mean non-price valuations (logit)
#     int = demand intercepts (linear, log-linear)
#     slopes = price coefficients (linear, log-linear)
#   Output is:
#     (unnamed) vector of market shares

quant_logit = function(p,alpha,xsi){
  return(  exp(xsi+p*alpha) / (1+sum(exp(xsi+p*alpha))) )
}


###################################################################
# The following series of functions provide demand derivatives for the four demand systems
#   Inputs include:
#     p = vector of prices
#     alpha = price coefficient (logit)
#     xsi = vector mean non-price valuations (logit)
#     int = demand intercepts (linear, log-linear)
#     slopes = price coefficients (linear, log-linear)
#   Output is:
#     (unnamed) matrix of demand derivaties
#   In dqdp, entry (i,j) is the derivative of qi with respect to pj

dqdp_logit = function(p,alpha,xsi){
  s = quant_logit(p,alpha,xsi)
  temp = -tcrossprod(s,s)
  diag(temp) = s*(1-s)
  return(alpha*temp)
}



###################################################################
# Function provides the value of the f function that characterizes the FOCs
# Independent of functional form of demand system
# The value of f is zero when prices are in Nash Bertrand equilbrium
#   Inputs include:
#     p = vector of prices
#     c = vector of costs
#     dqdp = matrix of demand derivatives
#     owner = ownership matrix

f.foc = function(p,c,q,dqdp,owner){
  return (-p+c-solve(owner*dqdp)%*%q)
}

###################################################################
# Function provides the value of the f function for Logit Demand
# Calls *f.foc* once inputs are arranged
# The value returned is zero when prices are in Nash Bertrand equilibrium
# Inputs include:
#   p = vector of (candidate) prices
#   c = vector of costs
#   alpha = price coefficient
#   xsi = vector of mean non-price valuations
#   fNum = vector assigning products to firms

f.foc_logit = function(p,c,alpha,xsi,fNum){
  
  #Obtaining an ownership matrix
  owner = f.owner(fNum)
  
  #Shares and derivatives
  s = quant_logit(p=p,alpha=alpha,xsi=xsi)
  dqdp = dqdp_logit(p=p,alpha=alpha,xsi=xsi)
  
  #Value of f function
  fVal = f.foc(p=p,c=c,q=s,dqdp=dqdp,owner=owner)
  
  return(fVal)
}




