#####################################################################################
# This library contains functions for supply side inference after estimating a 
# multinomial logit demand model.  The included functions are:
# f.logit_dqdp - calculates the price derivatives and elasticities for a particular market
#####################################################################################

#####################################################################################
# f.logit_dqdp calculates the price derivatives and elasticities for a particular market
#   Inputs include:
#     pcoef = price coefficient
#     price = prices in market
#     share = unconditional choice probabilities
#   Output includes vectors:
#     dqdp = price derivatives
#     elas = elasticities
#####################################################################################

f.logit_dqdp <- function(pcoef,price,share){
  
  nprods <- length(share)
  
  # Cross-price derivatives
  s_mat <- matrix(share,nprods,nprods)
  dqdp <- -pcoef*s_mat*t(s_mat)
  
  # Own price derivatives
  diag(dqdp) <- pcoef*share*(1-share)
  
  # Elasticities
  p_mat <- matrix(price,nprods,nprods)
  elas <- dqdp*t(p_mat)/s_mat
  
  return(list(dqdp=dqdp,elas=elas))
}