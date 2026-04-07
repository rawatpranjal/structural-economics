# Value Function Iteration with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# PARAMETERS

## horizon
T = 50

## preferences
risk_aver = 2
beta = 0.97

## returns
r = 0.05
R = 1+r

## income
y = np.zeros(T)
y[0:30] = range(1,31)
y[30:50] = 5

## rescale income so that average income = 1
y = y/sum(y)

## asset grids
na = 1000
amax = 5
borrow_lim = -0.10
agrid_par = 1 # 1 for linear, 0 for L-shaped

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1

# SET UP GRIDS

## assets
agrid = np.linspace(0,1,na)
agrid = agrid**(1/agrid_par)
agrid = borrow_lim + (amax-borrow_lim)*agrid

## put explicit point at a=0
agrid[agrid==np.min(abs(agrid-0))] = 0

# UTILITY FUNCTION

if risk_aver==1:
    u = lambda c: np.log(c)
else:
    u = lambda c: (c**(1-risk_aver)-1)/(1-risk_aver)

u1 = lambda c: c**(-risk_aver)

# INITIALIZE ARRAYS

V = np.zeros((na,T))
con = np.zeros((na,T))
sav = np.zeros((na,T))
savind = np.zeros((na,T), dtype=int)

# DECISIONS AT t=T

savind[:,T-1] = np.argwhere(agrid==0)
sav[:,T-1] = 0
con[:,T-1] = R*agrid + y[T-1] - sav[:,T-1]
V[:,T-1] = u(con[:,T-1])

# SOLVE VALUE FUNCTION

for it in range(T-2,-1,-1):
    if Display >=1:
        print('Solving at age: ' + str(it+1))

    # loop over assets
    for ia in range(0,na):   
        cash = R*agrid[ia] + y[it]
        Vchoice = u(np.maximum(cash-agrid,1.0e-10)) + beta*V[:,it+1]
        V[ia,it] = np.max(Vchoice)
        savind[ia,it] = np.argmax(Vchoice)
        sav[ia,it] = agrid[savind[ia,it]]
        con[ia,it] = cash - sav[ia,it]

# SIMULATE
if DoSimulate == 1:
    aindsim = np.zeros(T+1, dtype=int)
    
    ## initial assets: uniform on [borrow_lim, amax]
    ainitial = 0
    
    ## allocate to nearest point on agrid
    aindsim[0] = interp1d(agrid,range(1,na+1),'nearest')(ainitial)
        
    ## simulate forward
    for it in range(0,T):
        print(' Simulating, time period ' + str(it+1))

        ## asset choice
        aindsim[it+1] = savind[aindsim[it],it]
    
    ## assign actual asset and income values
    asim = agrid[aindsim]
    csim = R*asim[0:T] + y - asim[1:T+1]

# MAKE PLOTS
if MakePlots==1:
    
    ## consumption and income path
    plt.plot(range(1,51),y,'k-',linewidth=1,label='Income')
    plt.plot(range(1,51),csim,'r--',linewidth=1,label = 'Consumption')
    plt.grid()
    plt.title('Income and Consumption')
    plt.legend()
    plt.show()

    ## wealth path function
    plt.plot(range(0,51),asim,'b-',linewidth=1)
    plt.plot(agrid,np.zeros(na),'k',linewidth=0.5)
    plt.grid()
    plt.title('Wealth')
    plt.show()