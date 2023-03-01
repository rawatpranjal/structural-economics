

################################################################################
# These two functions evaluate profitability of the retailer given
# an input price vector and a parameterized logit demand system.
#
# - retail() obtains the share, profit, and FOC value (equals zeros at solution).
#   It applies the logit/Bertrand simplification that FOC: p = c - 1/(a*(1-s))
#
# - retail_wrapper() considers a candidate markup and returns FOC value. It 
#   applies the logit/Bertrand trick that of equal markups across products.
#
# Inputs:
#   - m = scalar common markup (wrapper only)
#   - p = price vector (retail only)
#   - xi = mean consumer valuation for non-price characteristics
#   - apar = logit price parameter
################################################################################

retail_wrapper <- function(m,w,xi,apar){
  p = m + w
  res <- retail(p=p,w=w,xi=xi,apar=apar)
  return(res$fval[1])
}

retail <- function(p,w,xi,apar){
  num <- exp(xi + apar*p)
  den <- 1+sum(num)
  s <- num / den
  ret_prof <- (p - w)*s
  ret_totprof <- sum(ret_prof)
  fval <- p - w + 1/(apar*(1-sum(s)))
  return(list(p=p,s=s,ret_prof=ret_prof,ret_totprof=ret_totprof,fval=fval))
}



################################################################################
# These two functions evaluate Nash-in-Nash bargaining between a retailer and 
# three brands.
#
# - nashprod() gets, for any input prices, the Nash Product associated with each  
#   brand, the derivative of brand and retail profit WRT the input price, and 
#   the derivative of the Nash product WRT to input price. The last object  
#   equals zero at a maximum.
#
# - nashprod_wrap() evaluates candidate input prices given all qualities, the
#   demand parameters, and the bargaining parameters. It returns the Nash 
#   product derivatives for use in optimization.
#
# Inputs:
#   - w = vector of all (candidate) input prices
#   - xi = mean consumer valuation for non-price characteristics
#   - apar = logit price parameter
#   - theta = vector or scalar with bargaining parameter(s)
################################################################################

nashinnash_wrap <- function(w,xi,apar,theta){

  nnp_eval <- nashinnash(w=w,xi=xi,apar=apar,theta=theta) 
  return(nnp_eval$dnashprod)
}

nashinnash <- function(w,xi,apar,theta){
  
  # Computing profit-maximizing retail markup and prices
  mstar <- uniroot(retail_wrapper,c(0,10),tol=1e-14,w=w,xi=xi,apar=apar)$root
  pstar <- mstar + w  
  
  # Getting outcomes for the retailer and the brands  
  res <- retail(p=pstar,w=w,xi=xi,apar=apar) 
  res$brnd_prof <- res$s*w
  
  # shell files and loop
  nashprod <- dnashprod <- dpb_dw <- dpr_dw <- rep(0,3)
  
  for (i in 1:length(w)){

    # Obtain the retailer's outside option: profit if product 1 not available
    w2 <- w[-i]
    xi2 <- xi[-i]
    mstar2 <- uniroot(retail_wrapper,c(0,10),tol=1e-14,w=w2,xi=xi2,apar=apar)$root
    pstar2 <- mstar2 + w2
    res2 <- retail(p=pstar2,w=w2,xi=xi2,apar=apar) 
    
    # Calculate the Nash product  
    np_term1 <- (res$brnd_prof[i])^theta[i]
    np_term2 <- (res$ret_totprof-res2$ret_totprof)^(1-theta[i])
    nashprod[i] <- np_term1 * np_term2
    
    ###################################################################
    # Obtaining the derivative of the Nash product wrt the input price
    ###################################################################
    
    # Perturbing input price 
    w_hi <- w_lo <- w
    w_hi[i] <- w[i] + 1e-4
    w_lo[i] <- w[i] - 1e-4
    
    # Recalculating markup, prices, profit with higher input price
    mstar_hi <- uniroot(retail_wrapper,c(0,10),tol=1e-14,w=w_hi,xi=xi,apar=apar)$root
    pstar_hi <- mstar_hi + w_hi  
    res_hi <- retail(p=pstar_hi,w=w_hi,xi=xi,apar=apar) 
    res_hi$brnd_prof <- res_hi$s*w_hi
    
    # Recalculating markup, prices, profit with lower input price
    mstar_lo <- uniroot(retail_wrapper,c(0,10),tol=1e-14,w=w_lo,xi=xi,apar=apar)$root
    pstar_lo <- mstar_lo + w_lo  
    res_lo <- retail(p=pstar_lo,w=w_lo,xi=xi,apar=apar) 
    res_lo$brnd_prof <- res_lo$s*w_lo
    
    # Double-sided numeric derivatives for how input prices affect profit
    dpb_dw[i] <- (res_hi$brnd_prof[i] - res_lo$brnd_prof[i] ) / (2*1e-4)
    dpr_dw[i] <- (res_hi$ret_totprof - res_lo$ret_totprof ) / (2*1e-4)  
    
    # Derivative of the Nash product WRT the input price
    dnp_term1 <- theta[i]*(res$brnd_prof[i])^(theta[i]-i) * dpb_dw[i] * np_term2
    dnp_term2 <- np_term1 * (1-theta[i])*(res$ret_totprof-res2$ret_totprof)^(-theta[i])*dpr_dw[i] 
    dnashprod[i] = dnp_term1 + dnp_term2    
  
  }
  
  
  return(list(nashprod=nashprod,dnashprod=dnashprod,
              res=res,dpb_dw=dpb_dw,dpr_dw))
}









