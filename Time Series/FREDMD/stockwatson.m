PATH = '/Users/pranjal/Desktop/Quantitative-Macroeconomics/Time Series/FREDMD/macro-history.xlsx';
df = readtable(PATH, sheet = 'Data');

df((df.country == 'USA'),:)