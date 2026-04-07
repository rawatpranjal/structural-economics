################################################################################
# This program is for Coding Exercise 4    
# 
# Builds off the Barkley code. From that, I have removed the "S" variable that
# allowed for buses of different types (simple=better). I also have rewritten 
# all the code in R such that it follows the algorithms outlined in the 
# Gretchen Sileo lecture slides. There the main difference is that Sileo 
# expresses things mainly with conditional value functions whereas the Barkley 
# code works mainly with unconditional (ex ante) value functions. Also, in 
# implementing CCP estimation, Barkley applies the newer IV approach of 
# Kalouptsidi et al (2021), where I have stuck with a Hotz-Miller approach.
#
# Estimation recovers the structural parameters used to construct the data.
#
# NHM 3-18-2023
#
# Big assists from Ryan Mansley, Tianshi Mu, and Gretchen Sileo
# 
################################################################################

rm(list = ls())  # Clear the workspace
set.seed(12)     # Set the random seed


#Set working directory
#path <- "C:/Users/nhm27/Dropbox/Teaching Materials/ECON 631 (Empirical IO)/2023/Coding Exercise 4"
path <- "/Users/pranjal/Desktop/Structural-Economics/industrial-organization/7_dynamic_discrete_choice"

setwd(path)

#Read the relevant libraries
source("dynamics_functions.R")
library(ggplot2)
library(dplyr)

options(scipen = 999)  

#########################################################################
# Defining the states: mileage
# - x_min and x_max give the min and max mileage is 0 and the max is 15
# - delta_x is for discretizing the space
# - x_len gives the number of discrete states, which is 301
# - x is a vector of the states
#########################################################################

x_min <- 0.0
x_max <- 15.0
delta_x <- 0.05
x_len <- (x_max - x_min) / delta_x + 1
x <- seq(x_min, x_max, by = delta_x)

#########################################################################
# Defining the state transitions
# - F1, F2 has starting state as row i ; ending state as column k
# - Builds in assumptions on how these transitions occur
# - F2 is the transition that occurs **without** replacement
# - F1 is the transition that occurs **with** replacement and is simpler
# - F2b is like F2 but it is the *CDF* not the PDF of the transition distribution
# - Saving Euler's constant for now
#########################################################################

x_tday <- matrix(rep(x, each = x_len), nrow = x_len)
x_next <- t(x_tday)
f <- (x_next >= x_tday) * exp(- (x_next - x_tday)) * (1 - exp(-delta_x))
f <- t(f)
f[, ncol(f)] <- 1 - rowSums(f[, 1:(ncol(f) - 1)])
F2 <- f
F2b <- t(apply(f, 1, cumsum))
F1 <- matrix(0, nrow = nrow(f), ncol = ncol(f))
F1[, 1] <- 1

#########################################################################
# Parameterization
# - beta: Discount factor 
# - theta: theta[1] = base utility of "not repair" (+), 
#          theta[2] = utility of "not repair" decreases with mileage (-) 
# - T: let this play out over T time periods
# - N: number of buses 
#########################################################################

beta <- 0.9
theta <- c(2.00, -0.15)  
T <- 30
X <- x
N <- 2000

#test
flow = cbind( rep(0,length(X)), theta[1] + theta[2]*X )
condval <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)$condval
ccp <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)$ccp
condval[1:10,]

#########################################################################
# Generating data and plotting for Bus 1 for confirmation
# - N buses observed over T periods each 
# - Check that average CCP corresponds to average decision
# - Plot data for Bus 1, t=1,...15, to see whether it is reasonable
#########################################################################

busdata <- generate_data(N=N,T=T,F1=F1,F2=F2,F2b=F2b,X=X,theta=theta,beta=beta)

data_t <- busdata$data_t
data_x <- busdata$data_x
data_d <- busdata$data_d
data_x_index <- busdata$data_x_index


# Average CCP vs. empirical repair percentage [EPR = -(ave-2)) ]
# These line up ... **must exclude t=T as no decision is recorded there**
emp_ccp <- matrix(ccp[data_x_index], ncol=ncol(data_x_index))
ave_ccp <- mean(emp_ccp[, -ncol(emp_ccp)])
erp <- -(mean(data_d[, -ncol(data_d)])-2)
c(ave_ccp,erp) 


# Plot the data for Bus 1
data_to_plot <- tibble(
  Time = data_t[1, 1:15],
  Mileage = data_x[1, 1:15],
  Replacement = data_x[1, 1:15] == 0
)

ggplot(data_to_plot, aes(x = Time, y = Mileage)) +
  geom_line(aes(color = "Mileage")) +
  geom_vline(aes(xintercept = Time, color = "Replacement"), 
             data = filter(data_to_plot, Replacement), 
             linetype = "dashed") +
  labs(x = "Time", y = "Mileage", title = "Mileage and replacement for bus #1",
       color = "Legend") +
  scale_x_continuous(breaks = seq(min(data_to_plot$Time), max(data_to_plot$Time), by = 1)) +
  scale_color_manual(values = c("Mileage" = "black", "Replacement" = "red")) +
  theme_minimal() +
  theme(panel.grid.major.x = element_line(size = 0.1, linetype = "solid", color = "gray"),
        panel.grid.major.y = element_line(size = 0.1, linetype = "solid", color = "gray"),
        panel.grid.minor.x = element_blank())

ggsave("bus_history_1.pdf", width = 6, height = 4, dpi = 300)



#########################################################################
# Estimating with Full Solution, Value Function Iteration (FIML)
#########################################################################

# Starting values
theta_cand <- theta

# Will impose bounds to avoid extreme parameters
theta_cand_lower <- c(1, -0.20)
theta_cand_upper <- c(3, -0.10)

# Converges to lower bounds
opt_FIML <- optim(theta_cand*0.9, f_FIML, busdata=busdata, F1=F1, F2=F2, X=X, beta=beta, 
               method = "L-BFGS-B", lower = theta_cand_lower, upper = theta_cand_upper)

print(opt_FIML)

# Evaluating objective function for visual on convergence
f_FIML(theta=theta*0.8,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta)
f_FIML(theta=theta*0.9,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta)
f_FIML(theta=theta*1.0,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta)
f_FIML(theta=theta*1.1,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta)
f_FIML(theta=theta*1.2,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta)




#########################################################################
# Estimating with CCPs
#########################################################################

# Prepare the data for logit regression
logitdata <- as.data.frame(cbind((c(data_d)==1), c(data_x), c(data_x)^2))
names(logitdata) <- c("repair","mileage","mileage2")

# Estimate the logit model
logitres <- glm(repair ~ mileage + mileage2, family = binomial(link = "logit"), data = logitdata)
logitpar <- logitres$coef

# Predicted values for each state
#temp <- logitpar[1] + logitpar[2]*X 
temp <- logitpar[1] + logitpar[2]*X + logitpar[3]*X^2
ccp_hat <- exp(temp) / (1+exp(temp))

# True CCPs
flow = cbind( rep(0,length(X)), theta[1] +theta[2]*(0:(length(X)-1)) )
ccp_tru <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)$ccp

# Scatter--good until right at the end
plot(ccp_tru,ccp_hat)

# Starting values
theta_cand <- theta

# Will impose bounds to avoid extreme parameters
theta_cand_lower <- c(1, -0.20)
theta_cand_upper <- c(3, -0.10)

# Converges to lower bounds
opt_CCP <- optim(theta_cand, f_CCP_EST, busdata=busdata, F1=F1, F2=F2, X=X, beta=beta, ccp_hat=ccp_hat,
             method = "L-BFGS-B", lower = theta_cand_lower, upper = theta_cand_upper)

print(opt_CCP)

# Evaluating objective function for visual on convergence
f_CCP_EST(theta_cand=theta*0.8,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta,ccp_hat=ccp_hat)
f_CCP_EST(theta_cand=theta*0.9,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta,ccp_hat=ccp_hat)
f_CCP_EST(theta_cand=theta*1.0,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta,ccp_hat=ccp_hat)
f_CCP_EST(theta_cand=theta*1.1,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta,ccp_hat=ccp_hat)
f_CCP_EST(theta_cand=theta*1.2,busdata=busdata,F1=F1,F2=F2,X=X,beta=beta,ccp_hat=ccp_hat)





