#####################################################################################
# This library contains functions for estimating a random coefficient logit demand 
# model and performing supply side inference.  The included functions are:
# f.mu - calculates the consumer specific part of the utility function
# f.rcl_indsh - provides *consumer-specific* choice probabilities
# f.rcl_combshr - provides implied market shares
# f.contrMap - finds the mean valuations that let model match observed shares
# f.2sls - estimates parameters for the model y = x %*% theta1 + u via 2SLS
# f.mom - calculates the moments for the RCL model (z'xi)
# f.gmmobj - evaluates the gmm objective function
# f.mom_deriv_num - numerically calculates the derivatives of the moments wrt theta2
# f.var_cov_gmm - calculates the variance/covariance matrix of estimated parameters
# f.rcl_dqdp - calculates the price derivatives and elasticities for a particular market
# f.owner - converts firm ID vector into ownership matrix
# f.cost - calculates costs via bertrand FOCs for a single market
# f.foc - evaluates the FOCs for each product in a market, used to find optimal prices in counterfactuals
#####################################################################################

#####################################################################################
# f.mu calculates the consumer specific part of the utility function for a set of
# simulated consumers. 
#   Inputs include:
#     pipars  = matrix which maps demographics into coefficients
#     demos = simulated demographics/consumers
#     xlin = variables that enter linearly into utility function (constant, price,
#     characteristics, etc)
#   Output includes vectors:
#     mu  = vector of consumer-specific deviation from delta
#####################################################################################

f.mu <- function(pipars,demos,xlin){
  
  mu <- xlin%*%t(demos%*%pipars)
  
  return(mu)
}


#####################################################################################
# f.rcl_indshr provides *consumer-specific* choice probabilities for the RCL 
#   Inputs include:
#     delta = vector of mean valuations
#     mu    = matrix of consumer-specific deviations from delta
#     mktid  = market identifier
#   Output includes vectors:
#     sharei = unconditional choice probabilities 
#####################################################################################

f.rcl_indsh <- function(delta,mu,mktid){
	print(delta[1:5])
	print(mu[1:5])
  print(size(delta+mu))
  eg <- exp(delta+mu)
  print(eg[1:5])
  #print(eg)
  temp <- aggregate(eg,by=list(Category=mktid),FUN=sum)
  #print(temp)
  temp2 <- 1+data.matrix(temp[,-1])
  #print(temp2)
  denoms <- matrix(temp2[mktid,],dim(mu))
  #print(denoms)
  sharei <- eg / denoms
  
  return(sharei)
}

#####################################################################################
# f.rcl_combshr provides implied market shares for the RCL. Constructed by
# integrating over consumer-specific choice probabilities.
#   Inputs include:
#     delta = vector of mean valuations
#     mu    = matrix of consumer-specific deviations from delta
#     mktid  = market identifier
#   Output includes vectors:
#     share = unconditional choice probabilities 
#####################################################################################

f.rcl_combshr <- function(delta,mu,mktid){
  
  sharei <- f.rcl_indsh(delta,mu,mktid)
  share <- rowMeans(sharei)
  
  return(share)
}  

#####################################################################################
# f.contrMap finds the mean valuations that let model match observed shares.
# Basically applies the Nevo (2001) contraction mapping code, 
# with the convergence speed damped following Grigolon and verboven (2014)
# See also MW (2017), appendix D.2.
#   Inputs include:
#     mu  = vector of consumer-specific deviation from delta
#     idata = market data, including on shares
#   Output includes vectors:
#     delta = vector of mean valuations that match shares
#     cm_count = number of contraction mapping iterations
#     cm_loss = max deviation between model and share data
#####################################################################################

f.contrMap <- function(mu,idata){
  
  # Starting values based on NL version of the model 
  delta0 <- log(idata$share) - log(idata$s0)
  mval0 <- mvalold <- delta0   
  
  # Setting values for stopping criteria
  cm_loss <- cm_count <- 1
  
  # Applying contraction mapping; 
  # Necessary to damp speed with RCNL (Grigolon and Verboven (2014))
  # Applying tight convergence criterion
  while (cm_loss > 1e-12 & cm_count < 1000){
    cand_share <- f.rcl_combshr(mvalold,mu,idata$mktid)
    mvalnew <- mvalold + log(idata$share) - log(cand_share)
    cm_loss <- max(abs(mvalnew - mvalold))
    cm_count <- cm_count + 1
    mvalold <- mvalnew
  }
  delta <- mvalnew
  
  return(list(delta=delta,cm_count=cm_count,cm_loss=cm_loss))
}

#####################################################################################
# f.2sls estimates parameters for the model y = x %*% theta1 + u via two stage least 
# squares regression.
#   Inputs include:
#     y = independent variable
#     x = regressors
#     z = instruments
#   Output includes vectors:
#     theta1  = parameters
#     res = estimated residuals
#####################################################################################

f.2sls <- function(x,y,z){
  
  zx <- t(z)%*%x
  zzi <- solve(t(z)%*%z)
  zy <- t(z)%*%y
  theta1 <- solve(t(zx)%*%zzi%*%zx)%*%(t(zx)%*%zzi%*%zy)
  res <- y-x%*%theta1
  
  return(list(theta1=theta1,res=res))
}

#####################################################################################
# f.mom calculates the moments for the RCL model.
#   Inputs include:
#     theta2  = parameters
#     indata = market data, including on shares
#     demos = simulated demographics/consumers
#     x = variables that enter linearly into utility function (constant, price,
#     characteristics, etc)
#     z = instruments
#   Output includes vectors:
#     g  = the moments as used by the objective function (z*xi summed over observations)
#     gi = the moments before summing over observations, used to calculate efficient weight matrix
#####################################################################################

f.mom <- function(theta2,indata,demos,x,z){
  
  pipars <- matrix(0,nrow=2,ncol=3)
  
  pipars[1,2] <- theta2[1]
  pipars[2,3] <- theta2[2]
  
  mu <- f.mu(pipars,demos,x)
  
  delta <- f.contrMap(mu,indata)
  
  xi <- f.2sls(x,delta$delta,z)$res
  
  g <- t(z)%*%xi
  
  gi <- z*matrix(rep(xi,length(g)),dim(z))
  
  return(list(g=g,gi=gi))
  
}

#####################################################################################
# f.gmmobj calculates the gmm objective function.  Calls f.mom to calculate moments.
#   Inputs include:
#     theta2  = parameters
#     indata = market data, including on shares
#     demos = simulated demographics/consumers
#     x = variables that enter linearly into utility function (constant, price,
#     characteristics, etc)
#     z = instruments
#     W_mat = weight matrix
#   Output includes vectors:
#     obj  = value of objective function
#####################################################################################

f.gmmobj <- function(theta2,indata,demos,x,z,W_mat){
  
  g <- f.mom(theta2,indata,demos,x,z)$g
  
  obj <- t(g)%*%W_mat%*%g
  
  return(obj[1])
}

#####################################################################################
# f.mom_deriv_num numerically calculates the derivatives of the moments wrt theta2.
#   Inputs include:
#     theta2  = parameters
#     indata = market data, including on shares
#     demos = simulated demographics/consumers
#     x = variables that enter linearly into utility function (constant, price,
#     characteristics, etc)
#     z = instruments
#     inc = distance to perturb parameters in calculating numerical derivatives
#   Output includes vectors:
#     dg_dtheta2  = jacobian of moments
#####################################################################################

f.mom_deriv_num <- function(theta2,indata,demos,x,z,inc=0.00001){
  
  g <- f.mom(theta2,indata,demos,x,z)$g
  
  #Get deviations in objective function for small changes in parameters
  dg_dtheta2 <- matrix(0,length(g),length(theta2))
  for (p in 1:length(theta2)){
    theta2_plus_eps <- theta2
    theta2_plus_eps[p] <- theta2_plus_eps[p]+inc/2
    g_plus_eps <- f.mom(theta2_plus_eps,indata,demos,x,z)$g
    theta2_minus_eps <- theta2
    theta2_minus_eps[p] <- theta2_minus_eps[p]-inc/2
    g_minus_eps <- f.mom(theta2_minus_eps,indata,demos,x,z)$g
    dg_dtheta2[,p] <- (g_plus_eps-g_minus_eps)/inc
  }
  
  return(dg_dtheta2)
}

#####################################################################################
# f.var_cov_gmm calculates the variance/cov of parameters estimated via gmm.
#   Inputs include:
#     G  = derivative of moments wrt BOTH theta1 and theta2
#     W_mat = weight matrix
#     eff_W_mat_inv = inverse of the efficient weight matrix
#   Output includes vectors:
#     var_cov  = variance/covariance matrix
#####################################################################################

f.var_cov_gmm <- function(G, W_mat, eff_W_mat_inv){
  
  A <- t(G)%*%W_mat%*%G
  B <- t(G)%*%W_mat%*%eff_W_mat_inv%*%W_mat%*%G
  var_cov <- inv(A)%*%B%*%inv(A)
  
  return(var_cov)
  
}

#####################################################################################
# f.rcl_dqdp calculates the price derivatives and elasticities for a particular market
#   Inputs include:
#     pcoefi = price coefficients for all simulated consumers (for a single market)
#     price = prices in market
#     sharei = unconditional choice probabilities
#     prodid = product IDs for market
#   Output includes vectors:
#     dqdp = price derivatives
#     elas = elasticities
#####################################################################################

f.rcl_dqdp <- function(pcoefi,price,sharei,prodid){
  
  nprods <- length(prodid)
  
  pcoefimat <- t(matrix(pcoefi, nrow=dim(sharei)[2], ncol=dim(sharei)[1]))
  price_mat <- matrix(price, nrow=dim(sharei)[1], ncol=dim(sharei)[2])
  
  # Own price derivatives
  owni <- pcoefimat*sharei*(1-sharei)
  own <- rowMeans(owni)
  
  # Cross-price derivatives
  cross <- matrix(0,length(own),length(own))
  cross_elas <- matrix(0,length(own),length(own))
  for (prod in prodid){
    temp <- sharei[prodid==prod,]
    temp2 <- t(matrix(temp,dim(sharei)[2],dim(sharei)[1]))
    cross[prodid==prod,] <- rowMeans(-pcoefimat*sharei*temp2)
  }
  
  # Put them together
  dqdp <- cross
  diag(dqdp) <- own
  
  # Elasticities
  s_mat <- matrix(rowMeans(sharei),nprods,nprods)
  p_mat <- matrix(price,nprods,nprods)
  elas <- dqdp*t(p_mat)/s_mat
  
  return(list(dqdp=dqdp,elas=elas))
}

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

#####################################################################################
# f.cost calculates costs via bertrand FOCs for a single market
#   Inputs include:
#     price = price vector
#     share = share vector
#     dqdp = price derivative matrix
#     owner = ownership/market matrix
#   Output includes vectors:
#     cost = cost vector
#####################################################################################

f.cost <- function(price,share,dqdp,owner){
  
  cost <- price + solve(owner*t(dqdp)) %*% share
  
  return(cost)
}

#####################################################################################
# f.foc evaluates the FOCs for each product in a market given a candidate price vector
# used to find equilibrium prices.
#   Inputs include:
#     price = price vector
#     pipars = matrix which maps demographics into coefficients
#     alpha = mean price coefficient
#     deltanp = mean valuation of each product with price component removed (x %*% beta + xi)
#     xnp = variables other than price that enter linearly into utility ('x')
#     cost = marginal cost vector
#     owner = ownership/market matrix
#     prodid = product id vector
#     demos = simulated demographics/consumers
#   Output includes vectors:
#     zero = value of FOC
#####################################################################################

f.foc <- function(price,pipars,alpha,deltanp,xnp,cost,owner,prodid,demos){
  
  # Adjusted mean valuations
  delta <- deltanp + alpha*as.vector(price)
  
  # Consumer-specific deviations (ai and mu), rcnl_functions.R
  ai <- as.vector(pipars[,1]%*%t(demos))
  mu <- f.mu(pipars,demos,cbind(price,xnp))
  
  # Price estimates, rcnl_functions.R
  pcoefi <- alpha + ai
  
  # Market shares, rcnl_functions.R
  sharei <- f.rcl_indsh(delta,mu,rep(1,length(delta)))
  share <- rowMeans(sharei)
  
  # Derivative of share with respect to price, rcnl_functions.R
  dqdp <- f.rcl_dqdp(pcoefi,price,sharei,prodid)$dqdp
  
  # Markups under Bertrand competition 
  markup <-   - solve(owner*t(dqdp)) %*% share
  markup <- as.vector(markup)
  
  # First order condition for prices to be optimized
  zero <- price - cost - markup
  
  return(zero)
}