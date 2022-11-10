# Aiyagari model
# Endogenous Grid Points with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Jan 2021

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

## production
deprec = 0.10
capshare = 0.4

## income risk: discretized N(mu,sigma^2)
mu_y = 1
sd_y = 0.2
ny = 5

## asset grids
na = 40
amax = 50
borrow_lim = 0
agrid_par = 0.5 # 1 for linear, 0 for L-shaped

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 50000
Tsim = 500

maxiter_KL = 70
tol_KL = 1.0e-5
step_KL = 0.005
rguess = 1/beta-1-0.001 # a bit lower than inverse of discount rate
KLratioguess = ((rguess + deprec)/capshare)**(1/(capshare-1))

# OPTIONS
Display = 1
MakePlots = 1

## which function to interpolation 
InterpCon = 0
InterpEMUC = 1

## tolerance for non-linear solver
TolX=1.0e-6

# UTILITY FUNCTION

if risk_aver==1:
    u = lambda c: np.log(c)
else:
    u = lambda c: (c**(1-risk_aver)-1)/(1-risk_aver)
    
u1 = lambda c: c**(-risk_aver)
u1inv = lambda u: u**(-1/risk_aver)

# DRAW RANDOM NUMBERS

np.random.seed(2021)
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

# SIMULATE LABOR EFFICIENCY REALIZATIONS
if Display>=1:
    print("Simulating labor efficiency realizations in advance")
yindsim = np.zeros((Nsim,Tsim), dtype=int)
    
for it in range(0,Tsim):

    ## income realization: note we vectorize simulations at once because
    ## of matlab, in other languages we would loop over individuals
    yindsim[yrand[:,it]<=ycumdist[0],it] = 0
    for iy in range(1,ny):
        yindsim[np.logical_and(yrand[:,it]>ycumdist[iy-1], yrand[:,it]<=ycumdist[iy]),it] = iy

ysim = ygrid[yindsim]

# ITERATE OVER KL RATIO
KLratio = KLratioguess

iterKL = 0
KLdiff = 1

while iterKL<=maxiter_KL and abs(KLdiff)>tol_KL:
    iterKL = iterKL + 1

    r = capshare*(KLratio**(capshare-1)) - deprec
    R = 1+r
    wage = (1-capshare)* (KLratio**capshare)

    ## rescale efficiency units of labor so that output = 1
    yscale = (KLratio**(-capshare))/(ygrid.T @ ydist)
    
    ## initialize consumption function in first iteration only
    if iterKL==1:
        conguess = np.zeros((na,ny))
        for iy in range(0,ny):
            conguess[:,iy] = r*agrid[:,0] + wage*yscale*ygrid[iy]
        con = conguess.copy()

    ## solve for policy functions with EGP
    
    Iter = 0
    cdiff = 1000
    while Iter<=max_iter and cdiff>tol_iter:
        Iter = Iter + 1
        sav = np.zeros((na,ny))
        
        conlast = con.copy()

        emuc = u1(conlast) @ ydist
        muc1 = beta*R*emuc
        con1 = u1inv(muc1)

        ## loop over income
        ass1 = np.zeros((na,ny))
        for iy in range(0,ny):

            ass1[:,iy] = ((con1 + agrid -wage*yscale*ygrid[iy])/R)[:,0]

            ## loop over current period ssets
            for ia in range(0,na): 
                if agrid[ia]<ass1[0,iy]: # borrowing constraint binds
                    sav[ia,iy] = borrow_lim
                else: # borrowing constraint does not bind
                    sav[ia,iy] = lininterp1(ass1[:,iy],agrid[:,0],agrid[ia])
            con[:,iy] = (R*agrid + wage*yscale*ygrid[iy])[:,0] - sav[:,iy]

        cdiff = np.max(abs(con-conlast))
        if Display>=2:
            print('Iteration no. ' + str(Iter), ' max con fn diff is ' + str(cdiff))

    ## simulate: start at assets from last interation
    if iterKL==1:
        asim = np.zeros((Nsim,Tsim))
    elif iterKL>1:
        # asim[:,0] = Ea.*ones(Nsim,1)
        asim[:,0] = asim[:,Tsim-1]
        
    ## create interpolating function
    savinterp = list()
    for iy in range(0,ny):
        savinterp.append(interp1d(agrid[:,0],sav[:,iy],'linear'))
    
    ## loop over time periods
    for it in range(0,Tsim):
        if Display>=2 and (it+1)%100==0:
            print("Simulating, time period " + str(it))
        
        ## asset choice
        if it < Tsim-1:
            for iy in range(0,ny):
                asim[yindsim[:,it]==iy,it+1] = savinterp[iy](asim[yindsim[:,it]==iy,it])
            
    ## assign actual labor income values
    labincsim = wage*yscale*ysim

    ## mean assets and efficiency units
    Ea = np.mean(asim[:,Tsim-1])
    L = yscale*np.mean(ysim[:,Tsim-1])
    
    KLrationew = (Ea/ L)[0,0]
    
    KLdiff = KLrationew/KLratio - 1
    if Display>=1:
        print("Equm iter " + str(iterKL) + ", r = " + str(r) + ", KL ratio: " + str(KLrationew) + " KL diff: " + str(KLdiff*100) + "%")

    KLratio = (1-step_KL)*KLratio + step_KL*KLrationew
    
# MAKE PLOTS
if MakePlots==1:
    
    ## consumption policy function
    plt.plot(agrid,con[:,0],'b-',label = 'Lowest income state')
    plt.plot(agrid,con[:,ny-1],'r-', label = 'Highest income state')
    plt.grid()
    plt.xlim((0,amax))
    plt.title('Consumption Policy Function')
    plt.legend()
    plt.show()

    ## savings policy function
    plt.plot(agrid,sav[:,0]-agrid[:,0],'b-')
    plt.plot(agrid,sav[:,ny-1]-agrid[:,0],'r-')
    plt.plot(agrid,np.zeros((na,1)),'k',linewidth=0.5)
    plt.grid()
    plt.xlim((0,amax))
    plt.title('Savings Policy Function (a''-a)')
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
    plt.ylim((0,2*Ea))
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