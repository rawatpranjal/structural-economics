# Value Function Iteration with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# PARAMETERS

## preferences
risk_aver = 2
beta = 0.95

## returns
r = 0.03
R = 1+r

## income
y = 1

## asset grids
na = 1000
amax = 20
borrow_lim = 0
agrid_par = 1 # 1 for linear, 0 for L-shaped

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 100
Tsim = 500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1

# SET UP GRIDS

## assets
agrid = np.linspace(0,1,na)
agrid = agrid**(1/agrid_par)
agrid = borrow_lim + (amax-borrow_lim)*agrid

# DRAW RANDOM NUMBERS

np.random.seed(2020)
arand = np.random.rand(Nsim)

# UTILITY FUNCTION

if risk_aver==1:
    u = lambda c: np.log(c)
else:
    u = lambda c: (c**(1-risk_aver)-1)/(1-risk_aver)

u1 = lambda c: c**(-risk_aver)

# INITIALIZE VALUE FUNCTION

Vguess = u(r*agrid+y)/(1-beta)

# ITERATE ON VALUE FUNCTION

V = Vguess.copy()

Vdiff = 1
Iter = 0

while Iter<=max_iter and Vdiff>tol_iter:
    Iter = Iter + 1
    Vlast = V.copy()
    V = np.zeros(na)
    sav = np.zeros(na)
    savind = np.zeros(na, dtype=int)
    con = np.zeros(na)
    
    ## loop over assets
    for ia in range(0,na):
        cash = R*agrid[ia] + y
        Vchoice = u(np.maximum(cash-agrid,1.0e-10)) + beta*Vlast         
        V[ia] = np.max(Vchoice)
        savind[ia] = np.argmax(Vchoice)
        sav[ia] = agrid[savind[ia]]
        con[ia] = cash - sav[ia]
    
    Vdiff = np.max(abs(V-Vlast))
    if Display >= 1:
        print('Iteration no. ' + str(Iter), ' max val fn diff is ' + str(Vdiff))

# SIMULATE
if DoSimulate==1:
    yindsim = np.zeros((Nsim,Tsim), dtype=int)
    aindsim = np.zeros((Nsim,Tsim), dtype=int)
    
    ## initial assets: uniform on [borrow_lim, amax]
    ainitial = borrow_lim + arand*(amax-borrow_lim)
    
    ## allocate to nearest point on agrid
    aindsim[:,0] = interp1d(agrid,range(1,na+1),'nearest')(ainitial)
    
    ## loop over time periods
    for it in range(0,Tsim):
        if Display >= 1 and (it+1)%100 == 0:
            print(' Simulating, time period ' + str(it+1))
        
        ## asset choice
        if it < Tsim-1:
            aindsim[:,it+1] = savind[aindsim[:,it]]

    ## assign actual asset and income values
    asim = agrid[aindsim]
    csim = R*asim[:,0:Tsim-1] + y - asim[:,1:Tsim]
    
# MAKE PLOTS
if MakePlots==1:
    
    ## consumption policy function
    plt.plot(agrid,con,'b-',linewidth=1)
    plt.grid()
    plt.xlim((0,amax))
    plt.title('Consumption Policy Function')
    plt.show()

    ## savings policy function
    plt.plot(agrid,sav-agrid,'b-',linewidth=1)
    plt.plot(agrid,np.zeros(na),'k',linewidth=0.5)
    plt.grid()
    plt.xlim((0,amax))
    plt.title('Savings Policy Function (a\'-a)')
    plt.show()
    
    ## nice zoom
    xlimits = (0,1)
    xlimind = np.ones(na, dtype=bool)
    if np.min(agrid) < xlimits[0]:
        xlimind = np.logical_and(xlimind,(agrid>=np.max(agrid[agrid<xlimits[0]])))
    elif np.min(agrid) > xlimits[1]:
        xlimind = 0
    if np.max(agrid) > xlimits[1]:
        xlimind = np.logical_and(xlimind,(agrid<=np.min(agrid[agrid>xlimits[1]])))
    elif np.max(agrid) < xlimits[0]:
        xlimind = 0

    ## consumption policy function: zoomed in
    plt.plot(agrid[xlimind],con[xlimind],'b-o',linewidth=2)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Consumption: Zoomed')
    plt.show()

    ## savings policy function: zoomed in
    plt.plot(agrid[xlimind],sav[xlimind]-agrid[xlimind],'b-o',linewidth=2)
    plt.plot(agrid,np.zeros((na,1)),'k',linewidth =0.5)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Savings: Zoomed (a\'-a)')
    plt.show()

    ## asset dynamics distribution
    plt.plot(range(0,Tsim),asim.T)
    plt.title('Asset Dynamics')
    plt.grid()
    plt.show()

    ## consumption dynamics distribution
    plt.plot(range(0,Tsim-1),csim.T)
    plt.title('Consumption Dynamics')
    plt.grid()
    plt.show()