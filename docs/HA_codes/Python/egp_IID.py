# Value Function Iteration with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from discrete_normal import discrete_normal
from lininterp1 import lininterp1

# PARAMETERS

## preferences
risk_aver = 2
beta = 0.95

## returns
r = 0.03
R = 1+r

## income risk: discretized N(mu,sigma^2)
mu_y = 1
sd_y = 0.2
ny = 5

## asset grids
na = 50
amax = 50
borrow_lim = 0
agrid_par = 0.5 # 1 for linear, 0 for L-shaped

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 50000
Tsim = 500

## mpc options
mpcamount1  = 1.0e-10 # approximate thoeretical mpc
mpcamount2  = 0.10 # one percent of average income: approx $500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1
ComputeMPC  = 1

# DRAW RANDOM NUMBERS

np.random.seed(2020)
yrand = np.random.rand(Nsim,Tsim)

# SET UP GRIDS

## assets
agrid = np.linspace(0,1,na).reshape(na,1)
agrid = agrid**(1/agrid_par)
agrid = borrow_lim + (amax-borrow_lim)*agrid

## income: disretize normal distribution
width = fsolve(lambda x: discrete_normal(ny,mu_y,sd_y,x)[0],2)
temp, ygrid, ydist = discrete_normal(ny,mu_y,sd_y,width)
ycumdist = np.cumsum(ydist)

# UTILITY FUNCTION

u = lambda c: (c**(1-risk_aver)-1)/(1-risk_aver) 
u1 = lambda c: c**(-risk_aver)
u1inv = lambda u: u**(-1/risk_aver)

# INITIALIZE CONSUMPTION FUNCTION

conguess = np.zeros((na,ny))
for iy in range(0,ny):
    conguess[:,iy] = (r*agrid+ygrid[iy])[:,0]

# ITERATE ON EULER EQUATION WITH ENDOGENOUS GRID POINTS

con = conguess.copy()

Iter = 0
cdiff = 1000

while Iter<=max_iter and cdiff>tol_iter:
    Iter = Iter + 1
    sav = np.zeros((na,ny))
    ass1 = np.zeros((na,ny))
    
    conlast = con.copy()
    
    emuc = u1(conlast) @ ydist
    muc1 = beta*R*emuc
    con1 = u1inv(muc1)
    
    ## loop over income
    for iy in range(0,ny):
        
        ass1[:,iy] = ((con1 + agrid - ygrid[iy])/R)[:,0]
        
        ## loop over current period ssets
        for ia in range(0,na): 
            if agrid[ia]<ass1[0,iy]: # borrowing constraint binds
                sav[ia,iy] = borrow_lim
            else: # borrowing constraint does not bind
                sav[ia,iy] = lininterp1(ass1[:,iy],agrid[:,0],agrid[ia])
                
        con[:,iy] = (R*agrid+ygrid[iy])[:,0] - sav[:,iy]

    cdiff = np.max(abs(con-conlast))
    if Display>=1:
        print('Iteration no. ' + str(Iter), ' max con fn diff is ' + str(cdiff))

# SIMULATE
if DoSimulate==1:
    yindsim = np.zeros((Nsim,Tsim), dtype=int)
    asim = np.zeros((Nsim,Tsim))
        
    ## create interpolating function
    savinterp = list()
    for iy in range(0,ny):
        savinterp.append(interp1d(agrid[:,0],sav[:,iy],'linear'))

    ## loop over time periods
    for it in range(0,Tsim):
        if Display >= 1 and (it+1)%100 == 0:
            print(' Simulating, time period ' + str(it+1))
        
        ## income realization: note we vectorize simulations at once because
        ## of matlab, in other languages we would loop over individuals
        yindsim[yrand[:,it]<=ycumdist[0],it] = 0
        for iy in range(1,ny):
            yindsim[np.logical_and(yrand[:,it]>ycumdist[iy-1], yrand[:,it]<=ycumdist[iy]),it] = iy
        
        ## asset choice
        if it < Tsim-1:
            for iy in range(0,ny):
                asim[yindsim[:,it]==iy,it+1] = savinterp[iy](asim[yindsim[:,it]==iy,it])

    ## assign actual income values
    ysim = ygrid[yindsim]

# MAKE PLOTS
if MakePlots==1:
    
    ## consumption policy function
    plt.plot(agrid,con[:,0],'b-',label = 'Lowest income state')
    plt.plot(agrid,con[:,ny-1],'r-', label = 'Highest income state')
    plt.grid()
    plt.xlim((0,amax))
    ### plt.title('Consumption Policy Function')
    plt.title('Consumption')
    plt.legend()
    plt.show()

    ## savings policy function
    plt.plot(agrid,sav[:,0]-agrid[:,0],'b-')
    plt.plot(agrid,sav[:,ny-1]-agrid[:,0],'r-')
    plt.plot(agrid,np.zeros((na,1)),'k',linewidth=0.5)
    plt.grid()
    plt.xlim((0,amax))
    ### plt.title('Savings Policy Function (a''-a)')
    plt.title('Savings')
    plt.show()

    ## nice zoom
    xlimits = (0,1)
    xlimind = np.ones(na, dtype=bool)
    if np.min(agrid) < xlimits[0]:
        xlimind = np.logical_and(xlimind,(agrid[:,0]>=np.max(agrid[agrid<xlimits[0]])))
    elif np.min(agrid) > xlimits[1]:
        xlimind = 0
    if np.max(agrid) > xlimits[1]:
        xlimind = np.logical_and(xlimind,(agrid[:,0]<=np.min(agrid[agrid>xlimits[1]])))
    elif np.max(agrid) < xlimits[0]:
        xlimind = 0

    ## consumption policy function: zoomed in
    plt.plot(agrid[xlimind],con[xlimind,0],'b-o',linewidth=2)
    plt.plot(agrid[xlimind],con[xlimind,ny-1],'r-o',linewidth=2)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Consumption: Zoomed')
    plt.show()

    ## savings policy function: zoomed in
    plt.plot(agrid[xlimind],sav[xlimind,0]-agrid[xlimind,0],'b-o',linewidth=2)
    plt.plot(agrid[xlimind],sav[xlimind,ny-1]-agrid[xlimind,0],'r-o',linewidth=2)
    plt.plot(agrid,np.zeros((na,1)),'k',linewidth =0.5)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Savings: Zoomed (a\'-a)')
    plt.show()

    ## income distribution
    plt.hist(ysim[:,Tsim-1],len(ygrid),facecolor=(0,0.5,0.5),edgecolor='blue')
    plt.ylabel('')
    plt.title('Income distribution')
    plt.show()

    ## asset distribution
    plt.hist(asim[:,Tsim-1],100,facecolor=(.7,.7,.7),edgecolor='black')
    plt.ylabel('')
    plt.title('Asset distribution')
    plt.show()

    ## convergence check
    plt.plot(range(0,Tsim),np.mean(asim,0),'k-',linewidth=1.5)
    plt.xlabel('Time Period')
    plt.title('Mean Asset Convergence')
    plt.show()

    ## asset distribution statistics
    aysim = asim[:,Tsim-1]/np.mean(ysim[:,Tsim-1])
    print('Mean assets: ' + str(np.mean(aysim)))
    print('Fraction borrowing constrained: ' + str(np.sum(aysim==borrow_lim)/Nsim * 100) + '%')
    print('10th Percentile: ' + str(np.quantile(aysim,.1)))
    print('50th Percentile: ' + str(np.quantile(aysim,.5)))
    print('90th Percentile: ' + str(np.quantile(aysim,.9)))
    print('99th Percentile: ' + str(np.quantile(aysim,.99)))

# COMPUTE MPCs
if ComputeMPC==1:
    
    ## theoretical mpc lower bound
    mpclim = R*((beta*R)**(-1/risk_aver))-1
        
    coninterp = list()
    mpc1 = np.zeros((na,ny))
    mpc2 = np.zeros((na,ny))
    
    for iy in range(0,ny):
        ## create interpolating function
        coninterp.append(interp1d(agrid[:,0],con[:,iy],'linear',fill_value='extrapolate'))
        
        mpc1[:,iy] = ( coninterp[iy](agrid[:,0]+mpcamount1) - con[:,iy] ) / mpcamount1
        mpc2[:,iy] = ( coninterp[iy](agrid[:,0]+mpcamount2) - con[:,iy] ) / mpcamount2

    ## mpc functions
    plt.plot(agrid,mpc1[:,0],'b-',linewidth=1,label='Lowest income state: amount 1')
    plt.plot(agrid,mpc2[:,0],'b--',linewidth=1,label='Lowest income state: amount 2')
    plt.plot(agrid,mpc1[:,ny-1],'r-',linewidth=1,label='Highest income state: amount 1')
    plt.plot(agrid,mpc2[:,ny-1],'r--',linewidth=1,label='Highest income state: amount 2')
    plt.plot(agrid,mpclim*np.ones(np.shape(agrid)),'k:',linewidth=1,label='Theoretical MPC limit = '+str(mpclim))
    plt.grid()
    plt.xlim((0,10))
    plt.title('MPC Function')
    plt.legend()
    plt.show()
    
    ## mpc distribution
    mpc1sim = np.zeros((Nsim,1))
    mpc2sim = np.zeros((Nsim,1))
    for iy in range(0,ny):
        mpc1sim[yindsim[:,Tsim-1]==iy,0] = ( coninterp[iy](asim[yindsim[:,Tsim-1]==iy,Tsim-1]+mpcamount1) - coninterp[iy](asim[yindsim[:,Tsim-1]==iy,Tsim-1]) ) / mpcamount1
        mpc2sim[yindsim[:,Tsim-1]==iy,0] = ( coninterp[iy](asim[yindsim[:,Tsim-1]==iy,Tsim-1]+mpcamount2) - coninterp[iy](asim[yindsim[:,Tsim-1]==iy,Tsim-1]) ) / mpcamount2
    
    plt.hist(mpc1sim,np.linspace(0,1.5,76),facecolor=(.7,.7,.7),edgecolor='black',linestyle='-')
    plt.grid()
    plt.ylabel('')
    plt.title('MPC distribution')
    
    # mpc distribution statistics
    print('Mean MPC amount 1: ' + str(np.mean(mpc1sim)))
    print('Mean MPC amount 2: ' + str(np.mean(mpc2sim)))
    plt.show()