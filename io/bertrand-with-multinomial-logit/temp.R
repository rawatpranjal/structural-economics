rm(list = ls())
path <- "/Users/pranjal/Desktop/Structural-Economics/industrial-organisation/bertrand-with-logit-demand"
setwd(path)

#############################################################################

rawdata <- list("shares"=c(0.15,0.15,0.3,0.3))
rawdata$margin <- 0.5
rawdata$prices <- c(1,1,1,1)
rawdata$fNum <- fNum <- c(1,2,3,4)
rawdata$mktP <- 1
rawdata$mktQ <- 1
print(rawdata)

#############################################################################

m=rawdata$margin
s=rawdata$shares
p=rawdata$prices
fNum=rawdata$fNum
C <- matrix(rep(fNum,length(fNum)),nrow=length(fNum)) 
owner = C==t(C)
num_prod <- length(s)
c1 = p[1]*(1-m)
temp = -tcrossprod(s,s)
diag(temp) = s*(1-s)
alpha = -1/((1-s[1])*(p[1]-c1))
dqdp = alpha*temp 
c = as.vector(p + solve(owner*dqdp)%*%s )
div <- t(matrix(s,num_prod,num_prod))*matrix(1/(1-s),num_prod,num_prod)
#div =  tcrossprod(s,1/(1-s))
diag(div)<- -1
xsi = log(s/(1-sum(s)))-alpha*p
type_j <- exp(xsi-alpha*c)
type <- aggregate(type_j,by=list(fNum),FUN=sum)$x  

# Check FOC  
f.foc = function(p,c,q,dqdp,owner){return (-p+c-solve(owner*dqdp)%*%q)}
f.foc(p,c,s,dqdp,owner)
  
f.foc_logit = function(p,c,alpha,xsi,fNum){
  owner = f.owner(fNum)
  s = quant_logit(p=p,alpha=alpha,xsi=xsi)
  dqdp = dqdp_logit(p=p,alpha=alpha,xsi=xsi)
  fVal = f.foc(p=p,c=c,q=s,dqdp=dqdp,owner=owner)
  return(fVal)
}

f.owner <- function(vec) {
  C <- matrix(rep(vec,length(vec)),nrow=length(vec))  
  return(C==t(C))
}

quant_logit = function(p,alpha,xsi){
  return(exp(xsi-p*alpha) / (1+sum(exp(xsi-p*alpha))) )
}

dqdp_logit = function(p,alpha,xsi){
  s = quant_logit(p,alpha,xsi)
  temp = -tcrossprod(s,s)
  diag(temp) = s*(1-s)
  return(alpha*temp)
}
#############################################################################
f.foc_logit(p,c,alpha,xsi,fNum)
  
