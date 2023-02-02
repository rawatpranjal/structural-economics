###########################################################################
# This program estimates a Bertrand/RCL model for a problem set for ECON 631
###########################################################################

rm(list = ls())

path = '/Users/pranjal/Desktop/Structural-Economics/industrial-organization/3_diff_prod_rand_logit/solutions'

setwd(path)

#Reading the relevant libraries
library(ivreg)
library(ggplot2)
library(pracma)
source("logit_functions.R")
source("rcl_functions.R")

#Replicability
set.seed(42, kind = NULL, normal.kind = NULL)


#############################################################################
# Getting Organized
#############################################################################

#Read in data
indata <- as.data.frame(read.csv("rcl_data_4.csv", header=TRUE))

#Summarize variables
summary(indata[,c('share','xvar','price','wvar')])

#Summarize number of products and firms per market
market_summary <- data.frame(mkt = unique(indata$mktid),
                             nprods = aggregate(prodid ~ mktid, indata, function(x) length(unique(x)))$prodid,
                             nfirms = aggregate(firmid ~ mktid, indata, function(x) length(unique(x)))$firmid)

summary(market_summary[,c('nprods','nfirms')])

#Calculate outside share
outside_shares <- aggregate(share ~ mktid, indata, sum)
indata$s0 <- 1-aggregate(share ~ mktid, indata, sum)$share[indata$mktid]

# Constructing candidate instruments
ivs <- indata[,c("mktid","prodid","firmid","xvar","wvar")]
for (mkt in unique(indata$mktid)){
  
  subset <- indata[indata$mktid==mkt,]
  
  # Price of product in other markets (Hausman IV)
  ivs[ivs$mktid==mkt,"hausman"] <- aggregate(price ~ prodid, indata[indata$mktid!=mkt,], mean)$price[subset$prodid]
  
  # Number of products owned by competing firm
  temp <- aggregate(xvar ~ firmid, subset, length)$xvar[subset$firmid]
  temp[is.na(temp)] <- 1
  ivs[ivs$mktid==mkt,"blp1"] <- length(subset$firmid) - temp     
      
  #Total number of products is co-linear in these data
  ivs[ivs$mktid==mkt,"blp2"] <- length(subset$firmid)
  
  # Sum of attributes in the market
  ivs[ivs$mktid==mkt,"blp3"] <- sum(subset$xvar)
  
  # Sum of attributes of products owned by competing firm
  temp <- aggregate(xvar ~ firmid, subset, sum)$xvar[subset$firmid]
  temp[is.na(temp)] <- 0
  ivs[ivs$mktid==mkt,"blp4"] <- sum(subset$xvar) - temp
  
  # Distance from all products
  temp <- subset$xvar - t(matrix(subset$xvar,length(subset$xvar),length(subset$xvar)))
  ivs[ivs$mktid==mkt,"diff1"] <- colSums(temp^2)
  
  # Distance from competing products
  ivs[ivs$mktid==mkt,"diff2"] <- colSums((1-f.owner(subset$firmid))*temp^2)

  # Sum of cost attributes in the market
  ivs[ivs$mktid==mkt,"cost2"]= sum(subset$wvar)
  
  # Sum of cost attributes of products owned by competing firms
  temp <- aggregate(wvar ~ firmid, subset, sum)$wvar[subset$firmid]
  temp[is.na(temp)] <- 0
  ivs[ivs$mktid==mkt,"cost3"] <- ivs[ivs$mktid==mkt,"cost2"] - temp
  
}



#############################################################################
# Logit section
#############################################################################

# Put IVs and data in one data frame
ivreg_data <- merge(indata,ivs)

# Create log(share/s0)
ivreg_data$y <- log(ivreg_data$share)-log(ivreg_data$s0)

# IV regression with only provided cost shifter, ivreg function is in the ivreg package
m_iv <- ivreg(y ~ xvar + price | xvar + wvar, data = ivreg_data)
summary(m_iv)

# Elasticities for market 1
alpha_logit <- m_iv$coefficients['price']
subset <- indata[indata$mktid==1,]
mkt1_logit_dqdp <- f.logit_dqdp(alpha_logit,subset$price,subset$share)
print(mkt1_logit_dqdp$elas)

# IV regression with hausman IV
m_iv <- ivreg(y ~ xvar + price | xvar + hausman, data = ivreg_data)
summary(m_iv)


#############################################################################
# Random Coefficient Logit section
#############################################################################

nobs <- length(indata$xvar)

# Consumer draws
ndemo <- 2
ndraw <- 100
draws <- matrix(rnorm(ndraw*ndemo),nrow=ndraw,ncol=ndemo)
draws <- draws -  t(matrix(colMeans(draws),ndemo,ndraw))

# Matrix of 'X's that are used in the inner 2SLS step
xlin <- cbind(indata$price,rep(1,nobs),indata$xvar)

# Matrix of Instruments, includes exogenous 'X's and actual instruments
z <- cbind(ivs$wvar,ivs$blp1,ivs$blp2,ivs$blp3,ivs$blp4,ivs$diff1,ivs$diff2,ivs$xvar,rep(1,nobs))

# Weighting matrix
W_mat <- solve(t(z)%*%z)

# Starting for the nonlinear parameters
theta2_guess <- c(1,1)

# Estimate the model
opt <- optim(theta2_guess,f.gmmobj,gr=NULL,indata,draws,xlin,z,W_mat,hessian = TRUE)

# Save theta2 estimates
theta2_hat <- opt$par

# Recover theta1 estimates

# Unpack theta2
pipars_hat <- matrix(0,nrow=2,ncol=3)
pipars_hat[1,2] <- theta2_hat[1]
pipars_hat[2,3] <- theta2_hat[2]

# Calculate mu's for simulated consumers at estimated theta2
mu_hat <- f.mu(pipars_hat,draws,xlin)

# Calculate mean valuations at estimated theta2
delta_hat <- f.contrMap(mu_hat,indata)

# Calculate theta1 via 2SLS
tsls_hat <- f.2sls(xlin,delta_hat$delta,z)
theta1_hat <- tsls_hat$theta1

# Calculate estimated residuals
xi_hat <- delta_hat$delta - xlin %*% theta1_hat

# Print paramter estimates
print(theta2_hat)
print(theta1_hat)

#Calculate SEs

# Moments at estimated theta2
mom_hat <- f.mom(theta2_hat,indata,draws,xlin,z)

# Calculate derivative of moments w.r.t theta2 (I did this numerically, but you can also derive an analytical expression)
dg_dtheta2 <- f.mom_deriv_num(theta2_hat,indata,draws,xlin,z,inc=0.00001)

# Efficient weight matrix
eff_W_mat_inv <- t(mom_hat$gi)%*%mom_hat$gi

# Combine derivative of moments w.r.t theta1 and theta2
G <- cbind(t(z)%*%xlin,dg_dtheta2)

# Do standard GMM math for variance
var_cov <- f.var_cov_gmm(G, W_mat, eff_W_mat_inv)

# Standard errors, note the order is the same as the order of derivatives in G (price, constant, xvar, constant_i, xvar_i)
SEs <- diag(var_cov)^0.5

# Elasticities
alpha_hat <- theta1_hat[1]
pcoefi_hat <- alpha_hat + t(draws%*%pipars_hat)[1,]
sharei_hat <- f.rcl_indsh(delta_hat$delta,mu_hat,indata$mktid)

mkt1_rcl_dqdp <- f.rcl_dqdp(pcoefi_hat,indata$price[indata$mktid==1],sharei_hat[indata$mktid==1,],indata$prodid[indata$mktid==1])

print(mkt1_rcl_dqdp$elas)

#############################################################################
# Supply Side section
#############################################################################

# Impute costs via FOC

indata$cost <- rep(0,length(indata$mktid))
for (mkt in unique(indata$mktid)){
  dqdp <- f.rcl_dqdp(pcoefi_hat,indata$price[indata$mktid==mkt],sharei_hat[indata$mktid==mkt,],indata$prodid[indata$mktid==mkt])
  owner <- f.owner(indata$firmid[indata$mktid==mkt])
  indata$cost[indata$mktid==mkt] <- f.cost(indata$price[indata$mktid==mkt],indata$share[indata$mktid==mkt],dqdp$dqdp,owner)
}

# Plot price vs cost and price vs xi

plot_data <- indata[,c('price','cost')]
plot_data$xi <- xi_hat

title <- sprintf('Price vs Cost')
ggplot() + 
  geom_point(data = plot_data, aes(x=cost, y=price)) + 
  ggtitle(title) + xlab('Cost') + ylab('Price')
filename <- sprintf('price_vs_cost.png')
ggsave(filename=filename, plot=last_plot(), device = "png" , dpi = 300)

title <- sprintf('Price vs xi')
ggplot() + 
  geom_point(data = plot_data, aes(x=xi, y=price)) + 
  ggtitle(title) + xlab('xi') + ylab('Price')
filename <- sprintf('price_vs_xi.png')
ggsave(filename=filename, plot=last_plot(), device = "png" , dpi = 300)


# Pass Through

deltanp_hat <- delta_hat$delta-alpha_hat*indata$price

market_summary$avg_price_inc <- rep(0,length(market_summary$mkt))
market_summary$hhi <- rep(0,length(market_summary$mkt))
for (mkt in unique(indata$mktid)){
  dqdp <- f.rcl_dqdp(pcoefi_hat[indata$mktid==mkt],indata$price[indata$mktid==mkt],sharei_hat[indata$mktid==mkt,],indata$prodid[indata$mktid==mkt])
  owner <- f.owner(indata$firmid[indata$mktid==mkt])
  cost_new <- indata$cost[indata$mktid==mkt] +0.5
  soln <- fsolve(f.foc,indata$price[indata$mktid==mkt],
                 pipars=pipars_hat,alpha=alpha_hat,deltanp=deltanp_hat[indata$mktid==mkt],xnp=xlin[indata$mktid==mkt,-1],
                 cost=cost_new,owner=owner,prodid=indata$prodid[indata$mktid==mkt],demos=draws)
  price_new <- soln$x
  market_summary$avg_price_inc[market_summary$mkt==mkt] <- mean(price_new-indata$price[indata$mktid==mkt])
  market_summary$hhi[market_summary$mkt==mkt] <- sum(indata$share[indata$mktid==mkt]^2)*10000
}

# Plot average price increase vs HHI

plot_data <- market_summary[,c('avg_price_inc','hhi')]

title <- sprintf('Average Price Increase vs HHI')
ggplot() + 
  geom_point(data = plot_data, aes(x=hhi, y=avg_price_inc)) + 
  ggtitle(title) + xlab('HHI') + ylab('Average Price Increase')
filename <- sprintf('avg_price_inc_vs_hhi.png')
ggsave(filename=filename, plot=last_plot(), device = "png" , dpi = 300)


# Cost function OLS regression

m_cost_ols <- lm(cost ~ wvar, data = indata)
summary(m_cost_ols)
  