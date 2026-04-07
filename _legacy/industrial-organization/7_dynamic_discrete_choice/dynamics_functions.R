

################################################################################
# This function performs a value function contraction mapping.
#
# Inputs include:
#   - flow: matrix with flow payoffs for each mileage (row) and action (column)
#   - F1: mileage transition if action=1 (repair)
#   - F2: mileage transition if action=2 (don't repair)
#   - X: possible mileage
#   - beta: discount factor
#
# Outputs include:
#   - condval: conditional values that satisfy contraction mapping
#   - iter: number of iterations done in contraction (curiosity)
#   - ccp: implied choice probability for action=1 (repair)
#
# This is used for FIML estimation and data generation
################################################################################

valuemap <- function(flow,F1,F2,X,beta){
  
  # Candidate value conditional value functions: rows are state; columns are choice
  v0 <- matrix(0,length(X),2)

  # Euler's constant. Not needed but allows condition value function to have
  # same units as the ex ante value function of Hotz-Miller so adding for 
  # better comparability 
  euler <- 0.5772
  
    
  # Solving the contraction mapping
  value_diff <- 1
  iter <- 0
  while ( value_diff > 1e-5 && iter<1000){
    
    # Implied inclusive value given state 
    ival <- log( rowSums(exp(v0))) + euler
    
    # Implied expectation of next period's value, by decision
    expval <- cbind(F1%*%ival,F2%*%ival)
    
    # Implied conditional values
    v1 <- flow + beta*expval
    
    # Maximum gap
    value_diff <- max(abs(v0-v1))
    
    # Update
    v0 <- v1
    
  }
  if (value_diff > 1e-5){print("Contraction Mapping Fails")}
  
  # Conditional Choice Probability (CCP): probability that choose repair (d=1)
  ccp <- 1 / (1 + exp(v0[,2] - v0[,1]))
  
  
  return(list(condval=v0,iter=iter,ccp=ccp))
}


################################################################################
# This function generates a sample of N buses observed over T years
#
# Inputs include:
#   - F1: mileage transition if action=1 (repair)
#   - F2: mileage transition if action=2 (don't repair)
#   - F2b: a CDF version of F2
#   - X: possible mileage
#   - theta: payoff parameters
#   - beta: discount factor
#
# Outputs include:
#   - data_x: mileages for buses (rows) and periods (columns)
#   - data_d: actions for buses (rows) and periods (columns); =1 for repair, =2 otherwise
#   - data_x_index: useful index
#
# This provides the data used in estimation
################################################################################

generate_data <- function(N,T,F1,F2,F2b,X,theta,beta) {
  
  initial_run <- 5
  x_index <- matrix(1, nrow = N, ncol = T + initial_run)
  
  # Decisions and mileage to be filled in
  d_sim <- matrix(0, nrow = N, ncol = T + initial_run)
  x_sim <- matrix(0, nrow = N, ncol = T + initial_run)
  
  # Random draws to translate probabilities into decisions and mileage transition
  draw_d <- matrix(runif(N * (T + initial_run)), nrow = N)
  draw_x <- matrix(runif(N * (T + initial_run)), nrow = N)
  
  # Flow utility.
  # - rows are state; columns are choice (1=repair, 2=no repair)
  flow = cbind( rep(0,length(X)), theta[1] +theta[2]*X )
  
  # Obtaining conditional value function and implied CCP.
  # - rows are state; columns are choice (1=repair, 2=no repair)
  solution <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)
  condval <- solution$condval
  ccp <- solution$ccp
  
  for (n in 1:N) {
    for (t in 1:(T + initial_run - 1)) {
      
      # Probability of "repair" and "don't repair"
      p1 <- ccp[x_index[n,t]]
      
      # Use random draws; =1 if "repair" and =2 if "don't repair"
      d_sim[n, t] <- (draw_d[n, t] > p1) + 1
      
      # If don't repair, transition to a new mileage state based on random draw
      x_index[n, t + 1] <- (d_sim[n, t] == 2) * sum(draw_x[n,t] > F2b[x_index[n, t], ]) + 1
      
      # Recording the implied mileage from that transition            
      x_sim[n, t + 1] <- X[x_index[n, t + 1]]
    }
  }
  
  data_x <- x_sim[, (initial_run + 1):(T + initial_run)]
  data_d <- d_sim[, (initial_run + 1):(T + initial_run)]
  data_t <- t(matrix(rep(1:T, N), nrow = T))
  data_x_index <- apply(data_x, 2, function(col) match(col, X))
  
  return(list(data_t = data_t, data_x = data_x, data_d = data_d, data_x_index = data_x_index))
}



################################################################################
# This function obtain the log-likelihood for FIML (Rust-style full information
# value function interation)
#
# Inputs include:
#   - theta_cand: candidate payoff parameters
#   - busdata: data on buses
#   - F1: mileage transition if action=1 (repair)
#   - F2: mileage transition if action=2 (don't repair)
#   - X: possible mileage
#   - beta: discount factor
#
# Outputs include:
#   - logL: (negative) log-likelihood function
#
################################################################################


f_FIML <- function(theta_cand,busdata,F1,F2,X,beta){
  
  #print(theta_cand)
  
  # Conditional flow utility at every state 
  flow <- cbind( rep(0,length(X)), theta_cand[1] +theta_cand[2]*X)
  
  # Contraction mapping: Conditional value functions, CCPs at every *state*
  solution <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)
  condval <- solution$condval
  ccp <- solution$ccp
  
  # CCPS for each *data point*
  p1 <- matrix(ccp[busdata$data_x_index], ncol=ncol(busdata$data_x_index))

  # Actual decisions for each point
  data_d <- busdata$data_d
  
  ## Coding check: similar probabilities of repair?
  #emp_ccp <- matrix(ccp[data_x_index], ncol=ncol(data_x_index))
  #ave_ccp <- mean(emp_ccp[, -ncol(emp_ccp)])
  #erp <- -(mean(data_d[, -ncol(data_d)])-2)
  #disp( c(ave_ccp,erp) )
  
  # Minimize this (negative) log-likelihood
  #  - Note the t=T data where we don't have decisions are zeroed out from this
  logL <- - sum (  (data_d==1)*log(p1) + (data_d==2)*log(1-p1))
  
  # Finishing
  return(logL)
  
}


################################################################################
# This function obtains the log-likelihood for Hotz-Miller-style CCP estimation
#
# Inputs include:
#   - theta_cand: candidate payoff parameters
#   - busdata: data on buses
#   - F1: mileage transition if action=1 (repair)
#   - F2: mileage transition if action=2 (don't repair)
#   - X: possible mileage
#   - beta: discount factor
#   - ccp_hat: pre-estimated CCPs 
#
# Outputs include:
#   - logL: (negative) log-likelihood function
#
################################################################################

f_CCP_EST <- function(theta_cand,busdata,F1,F2,X,beta,ccp_hat){

  #print(theta_cand)
  
  # Working on numerator--in state space--this is an N*1 vector
  euler <- 0.5772
  phat1 <- ccp_hat
  phat2 <- 1-phat1
  flow2 <- theta_cand[1] + theta_cand[2]*X
  numer <- phat1*(0-log(phat1)) + phat2*(flow2-log(phat2)) + euler
  
  # Working on denominator--in state space--this needs to be an N*N matrix
  eyeN <- diag(length(X))
  phat1_tile <- t(matrix(rep(phat1, each = length(phat1)), ncol = length(phat1)))
  phat2_tile <- 1 - phat1_tile
  denom <- eyeN - beta*phat1_tile*F1 - beta*phat2_tile*F2 
  
  # Implied ex ante value function--one value per state--N*1 vector
  nu <- solve(denom)%*%numer

  # Convert to implied CCPs
  term1 <- 0 + beta*F1%*%nu
  term2 <- flow2 + beta*F2%*%nu
  ccp_1 <- exp(term1) / (exp(term1) + exp(term2))

  ## Coding check: if ccp_hat = true ccp then these should be identical
  #cbind(ccp_1,ccp_hat)[1:10,]

  ## Coding check: The ex ante value is a bit larger than conditional values 
  # flow = cbind( rep(0,length(X)), theta_cand[1] +theta_cand[2]*X) 
  # solution <- valuemap(flow=flow,F1=F1,F2=F2,X=X,beta=beta)
  # condval <- solution$condval
  # ccp <- solution$ccp  
  # cbind(nu[1:10],condval[1:10,])    
    

  # Implied CCPs for each **data point**
  ccp_1b <- matrix(ccp_1[busdata$data_x_index], ncol=ncol(busdata$data_x_index)) 
  
  # Actual decisions for each point
  data_d <- busdata$data_d
  
  ## Coding check: return same probability of repair?
  #ave_ccp <- mean(ccp_1b[, -ncol(ccp_1b)])
  #erp <- -(mean(data_d[, -ncol(data_d)])-2)
  #disp( c(ave_ccp,erp) )
  
  
  # Minimize this (negative) log-likelihood
  #  - Note the t=T data where we don't have decisions are zeroed out from this  
  logL <- - sum (  (data_d==1)*log(ccp_1b) + (data_d==2)*log(1-ccp_1b))

  # Finishing
  return(logL)  
  
}




