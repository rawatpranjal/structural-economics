##########################################################################
# f.owner converts firm assignment into ownership matrix
#   Also can be used to dummy out observations from different markets
#   Inputs include:
#     vec= vector assigning products to firms (or markes)
#   Output includes:
#     (unnamed) ownership/market matrix
##########################################################################

f.owner <- function(vec) {
  
  C <- matrix(rep(vec,length(vec)),nrow=length(vec))  
  
  return(C==t(C))  
}

##########################################################################
# foc_partial provides evaluates first order conditions for profit
# maximization, potentially for a subset of products in a market. If
# uni==1 then returns sum of squared errors, otherwise returns the
# vector of errors (moments).
# Inputs include:
#   - p: vector of prices for evaluation
#   - popt: slots in p (and pfix) into the full price vector
#   - pfix: the prices to be held fixed
#   - alpha: price parameter
#   - sigma: nesting parameter
#   - pipars: parameters on random coefficients
#   - mc: vector of marginal costs
#   - group: vector assigning products to groups
#   - firmid: vector with firm identifier
#   - uni: toggle that determines output form
# Outputs include:
#   - zero = evaluation vector: 0s if FOC satisfied
#   - wprice = prices
#   - slist = implied market share list
##########################################################################

foc_partial_wrap <- function(p,popt,pfix,alpha,sigma,pipars,deltanp,mc,idata,uni){
  
  getres <- foc_partial(p,popt=popt,pfix=pfix,alpha=alpha,
                        sigma=sigma,pipars=pipars,deltanp=deltanp,
                        mc=mc,idata=idata)
  
  if (uni==1) {
    sendback <- sum(getres$zero^2)
  } else {
    sendback <- getres$zero }
  
  return(sendback)
}

foc_partial <- function(p,popt,pfix,alpha,sigma,pipars,deltanp,mc,idata){
  
  # Prices
  pvec <- rep(0,length(popt))
  pvec[popt==1] <- p
  pvec[popt==0] <- pfix[popt==0]
  
  # Adjusted mean valuations
  delta2 <- deltanp + alpha*pvec
  
  # Consumer-specific deviations (ai and mu), rcnl_functions.R
  ai2 <- as.vector(pipars[,1]%*%t(idata$demos))
  mu2 <- cbind(pvec,idata$x)%*%t(idata$demos%*%pipars)

  # Price estimates, rcnl_functions.R
  pcoefi <- alpha + ai2
  
  # Market shares, rcnl_functions.R
  indsh <- f.rcnl_indsh(delta=delta2,mu=mu2,sigma2=sigma,mktid=NA,mktindex=NA,loopflag=1)
  su <- rowMeans(indsh$sharei)
  
  # Derivative of share with respect to price, rcnl_functions.R
  dqdp <- f.rcnl_dqdp(pcoefi=pcoefi,sigma2=sigma,sharei=indsh$sharei,
                      scondi=indsh$scondi,sg_mat=indsh$sg_mat,idata=idata)$dqdp
  
  # Markups under Bertrand competition 
  markup <-   - solve(f.owner(idata$firmid[popt==1])*t(dqdp)) %*% su[popt==1]
  markup <- as.vector(markup)

  # First order condition for prices to be optimized
  zero <- p - mc[popt==1] - markup[popt==1]
  
  return(list(zero=zero,price=pvec,indsh=indsh,su=su,markup=markup))
}







