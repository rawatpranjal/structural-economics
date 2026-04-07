install.packages(imputeTS)
library(devtools)
install_github("SteffenMoritz/imputeTS")

# Get FRED Monthly Data
# Apply Missing Value Imputation
library(fbi)
filepath <- "/Users/pranjal/Desktop/Quantitative-Macroeconomics/Time Series/FREDMD/fred-md.csv"
data <- fredmd(filepath, date_start = NULL, date_end = NULL, transform = TRUE)
N <- ncol(data)
data_clean <- rm_outliers.fredmd(data)
col_na_prop <- apply(is.na(data_clean), 2, mean)
data_select <- data_clean[, (col_na_prop < 0.05)]
data_bal <- na.omit(data_select)
X_bal <- data_bal[,2:ncol(data_bal)]
rownames(X_bal) <- data_bal[,1]
out <- rpca(X_bal, kmax = 8, standardize = FALSE, tau = 0)
md <- data.frame(DATE=as.Date(rownames(X_bal), "%Y-%m-%d"),X_bal)
names(md) <- make.names(names(md), unique=TRUE)

# Get FRED Quarterly Data
# Apply Missing Value Imputation
library(fbi)
filepath <- "/Users/pranjal/Desktop/Quantitative-Macroeconomics/Time Series/FREDMD/fred-qd.csv"
data_clean <- fredqd(filepath, date_start = NULL, date_end = NULL, transform = TRUE)
N <- ncol(data_clean)
data_clean <- rm_outliers.fredqd(data)
col_na_prop <- apply(is.na(data_clean), 2, mean)
data_select <- data_clean[, (col_na_prop < 0.05)]
data_bal <- na.omit(data_select)
X_bal <- data_bal[,2:ncol(data_bal)]
rownames(X_bal) <- data_bal[,1]
out <- rpca(X_bal, kmax = 8, standardize = FALSE, tau = 0)
qd <- data.frame(DATE=as.Date(rownames(X_bal), "%Y-%m-%d"),X_bal)
names(qd) <- make.names(names(qd), unique=TRUE)

library(tidyverse)
library(ggplot2)
library(mFilter)
qd$GDP = log(cumprod(qd$GDPC1+1))

# GDP
gdp.bk = cffilter(qd$GDP, pl=16, pu=40, root=TRUE)
qd$GDP = na_ma(qd$GDP, k = 4, weighting = "simple")
plot(qd$DATE, gdp.bk$cycle, type ='l')
plot(qd$DATE, gdp.bk$trend, type ='l')

# CPI Inf
library(imputeTS)
qd$CPI = cumprod(cumprod(qd$CPIAUCSL+1))
qd$CPI = (log(qd$CPI)-log(lag(qd$CPI,4)))*100
qd$CPI = na_ma(qd$CPI, k = 4, weighting = "simple")
plot(qd$DATE, qd$CPI, type ='l')
cpi.bk = cffilter(qd$CPI, pl=16, pu=48, root=TRUE)
plot(qd$DATE, cpi.bk$cycle, type ='l')
plot(qd$DATE, cpi.bk$trend, type ='l')

# FEDS
qd$FEDFUNDS[1] = 5
qd$FED = cumsum(qd$FEDFUNDS)
qd$FED = na_ma(qd$FED, k = 4, weighting = "simple")
plot(qd$DATE, qd$FED, type ='l')
fed.bk = cffilter(qd$FED, pl=16, pu=48, root=TRUE)
plot(qd$DATE, fed.bk$cycle, type ='l')
plot(qd$DATE, fed.bk$trend, type ='l')

# Visualise
A = qd %>% ggplot(aes(x=DATE, y=cpi.bk$cycle))  + geom_line()
B = qd %>% ggplot(aes(x=DATE, y=fed.bk$cycle)) + geom_line()
C = qd %>% ggplot(aes(x=DATE, y=gdp.bk$cycle))  + geom_line()
library(gridExtra)
grid.arrange(A, B, C, ncol=1, nrow = 3)


# SVAR Model
# IRF & FEVD use Choleski Decomposition with FORMER variables in the list (e.g. cbind(INF, UNEMP, POL)) having contemporaneous effects on LATTER variables, while LATTER variables do not have contemporaneous effects on FORMER. 

INF = cpi.bk$cycle
POL = fed.bk$cycle
GDP = gdp.bk$cycle

library(vars)
VARselect(cbind(INF, GDP, POL), type = 'const')
model = VAR(cbind(INF, GDP, POL), p = 2, type = 'const')
summary(model)

# A, B Matrices for SVAR
amat <- diag(3)
diag(amat) <- NA
amat[2, 1] <- NA
amat[3, 1] <- NA
amat[3, 2] <- NA

model = SVAR(model, estmethod = "direct", Amat = amat, Bmat = NULL, hessian = TRUE, method = "BFGS")
summary(model)

# IRF
plot(irf(model, impulse = "POL", response = c('INF', 'UNEMP', 'POL'), boot = TRUE))
plot(irf(model, impulse = "UNEMP", response = c('INF', 'UNEMP','POL'), boot = TRUE))
plot(irf(model, impulse = "INF", response = c('INF', 'UNEMP','POL'), boot = TRUE))

# FEVD
plot(fevd(model, n.ahead = 3))
