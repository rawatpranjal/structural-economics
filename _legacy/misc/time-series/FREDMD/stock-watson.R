install.packages("fredr")
library(fredr)
fredr_set_key("b6d46f2d7c8e5cb6866cfa812c861cf9")
df = fredr("GDPC1")

CPI = fredr(series_id = "GDP",observation_start = as.Date("1990-01-01"),observation_end = as.Date("2000-01-01"),frequency = "q", units = "chg" )



UNEMP = fredr(series_id = "GDP",observation_start = as.Date("1990-01-01"),observation_end = as.Date("2000-01-01"),frequency = "q", units = "chg" )


PATH = "/Users/pranjal/Desktop/Quantitative-Macroeconomics/Time Series/FREDMD/macro-history.xlsx"
library(readxl)
library(tidyverse)
df = tibble(read_excel(PATH, sheet = 'Data'))
df = df %>% filter(country == 'USA') 
	
help(select)
