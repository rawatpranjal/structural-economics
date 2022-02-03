
# Get FRED Monthly Data
# Apply Missing Value Imputation
library(fbi)
filepath <- "/Users/pranjal/Desktop/Quantitative-Macroeconomics/Time Series/fred-md.csv"
data <- fredmd(filepath, date_start = NULL, date_end = NULL, transform = TRUE)
N <- ncol(data)
data_clean <- rm_outliers.fredmd(data)
col_na_prop <- apply(is.na(data_clean), 2, mean)
data_select <- data_clean[, (col_na_prop < 0.05)]
data_bal <- na.omit(data_select)
X_bal <- data_bal[,2:ncol(data_bal)]
rownames(X_bal) <- data_bal[,1]
out <- rpca(X_bal, kmax = 8, standardize = FALSE, tau = 0)
df <- data.frame(DATE=as.Date(rownames(X_bal), "%Y-%m-%d"),X_bal)
names(df) <- make.names(names(df), unique=TRUE)


# Describe variables
describe_md("BAA")

library(tidyverse)
library(ggplot2)
df = as_tibble(df)
df = df %>% filter(df$DATE>='1960-01-01')

df %>% ggplot(aes(x=DATE, y=RPI)) + geom_point() + geom_line()



new_names
df <- data.frame(DATE=as.Date(rownames(X_bal), "%Y/%m/%d"),X_bal)
