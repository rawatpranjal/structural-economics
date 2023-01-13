# Endogenous Grid Points with AR1 + IID Income
# Cash on Hand as State variable
# Includes NIT and discount factor heterogeneity
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from discrete_normal import discrete_normal
from lininterp1 import lininterp1
from ginicoeff import ginicoeff
from rouwenhorst import rouwenhorst

# PARAMETERS

## preferences
risk_aver = 1
beta = 0.955

## warm glow bequests: bequest_weight = 0 is accidental
bequest_weight = 0 # 0.07
bequest_luxury = 2 # 0.01
dieprob     = 0 # 1/50

## returns
r = 0.02
R = 1+r

## income risk: AR(1) + IID in logs
nyT = 5 # transitory component (not a state variable)
sd_logyT = 0.20 # relevant if nyT>1

nyP = 11 # 9 # persistent component
sd_logyP = 0.24
rho_logyP = 0.97

## cash on hand / savings grid
nx = 50
xmax = 40
xgrid_par = 0.4 # 1 for linear, 0 for L-shaped
borrow_lim = 0

## government
labtaxlow = 0 # proportional tax
labtaxhigh = 0 # additional tax on incomes above threshold
labtaxthreshpc = 0.99 # percentile of earnings distribution where high tax rate kicks in

savtax = 0.0001 # tax rate on savings
savtaxthresh = 0 # multiple of mean gross labor income


## discount factor shocks
nb = 1 # 1 or 2
betawidth = 0.065 # beta +/- beta width
betaswitch = 1/40 # 0

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 100000
Tsim = 300

## mpc options
mpcamount1 = 1.0e-10 # approximate thoeretical mpc
mpcamount2 = 0.10 # one percent of average income: approx $500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1
ComputeMPC = 1

## which function to interpolation 
InterpMUC = 1
InterpCon = 0 # sometimes useful to stop extrapolating to negative MUC


# DRAW RANDOM NUMBERS
np.random.seed(2020)
yPrand = np.random.rand(Nsim,Tsim)
yTrand = np.random.randn(Nsim,Tsim)
betarand = np.random.rand(Nsim,Tsim)
dierand = np.random.rand(Nsim,Tsim)

# INCOME GRIDS

## persistent income: rowenhurst
logyPgrid, yPtrans, yPdist = rouwenhorst(nyP, -0.5*sd_logyP**2, sd_logyP, rho_logyP)
yPgrid = np.exp(logyPgrid)
yPcumdist = np.cumsum(yPdist)
yPcumtrans = np.cumsum(yPtrans,axis=1)

## transitory income: disretize normal distribution
if nyT>1:
    width = fsolve(lambda x: discrete_normal(nyT,-0.5*sd_logyT**2,sd_logyT,x)[0],2)
    temp,logyTgrid,yTdist = discrete_normal(nyT,-0.5*sd_logyT**2,sd_logyT,width)
elif nyT==1:
    logyTgrid = 0
    yTdist = 1
yTgrid = np.exp(logyTgrid)

## gross labor income grid;
grosslabincgrid = np.zeros((nyP,nyT))
grosslabincdist = np.zeros((nyP,nyT))
for iyP in range(0,nyP):
    for iyT in range(0,nyT):
        grosslabincgrid[iyP,iyT] = yPgrid[iyP]*yTgrid[iyT]
        grosslabincdist[iyP,iyT] = yPdist[iyP]*yTdist[iyT]

## gross labor income distribution;
grosslabincgridvec = np.concatenate(grosslabincgrid).reshape(nyP*nyT,1)
grosslabincdistvec = np.concatenate(grosslabincdist).reshape(nyP*nyT,1)
temp = np.block([grosslabincgridvec, grosslabincdistvec])
temp = temp[np.argsort(temp[:,0])]
glincgridvec = temp[:,0]
glincdistvec = temp[:,1]
glincdistcumvec = np.cumsum(glincdistvec)

## labor taxes and transfers function
meangrosslabinc = np.sum(glincgridvec*glincdistvec)
totgrosslabinc = meangrosslabinc
labtaxthresh = lininterp1(glincdistcumvec,glincgridvec,labtaxthreshpc)
totgrosslabinchigh = np.sum(np.maximum(glincgridvec - labtaxthresh,0)*glincdistvec)

lumptransfer = labtaxlow*totgrosslabinc + labtaxhigh*totgrosslabinchigh

## net labor income
netlabincgrid = lumptransfer + (1-labtaxlow)*grosslabincgrid - labtaxhigh*np.maximum(grosslabincgrid- labtaxthresh,0)

# ASSET GRIDS

xgrid = np.zeros((nx,nyP))

## cash on hand grids: different min points for each value of (iyP)
for iyP in range(0,nyP):
    xgrid[:,iyP] = np.linspace(0,1,nx)
    xgrid[:,iyP] = xgrid[:,iyP]**(1/xgrid_par)
    xgrid[:,iyP] = borrow_lim + min(netlabincgrid[iyP,:]) + (xmax-borrow_lim)*xgrid[:,iyP]

## savings grid for EGP
ns = nx
sgrid = np.linspace(0,1,nx).reshape(nx,1)
sgrid = sgrid**(1/xgrid_par)
sgrid = borrow_lim + (xmax-borrow_lim)*sgrid

## discount factor grid
if nb==1:
    betagrid = np.array([[beta]])
    betadist = np.ones((1,1))
    betatrans = np.ones((1,1))
    betacumdist = np.ones((1,1))
    betacumtrans = np.ones((1,1))
elif nb==2:
    betagrid = np.array([[beta-betawidth],[beta+betawidth]])
    betadist = np.array([[0.5],[0.5]])
    betatrans = np.array([[1-betaswitch, betaswitch],[betaswitch, 1-betaswitch]]) # transitions on average once every 40 years
    betacumdist = np.cumsum(betadist)
    betacumtrans = np.cumsum(betatrans,1)
else:
    raise SystemExit('nb must be 1 or 2')
    
# UTILITY FUNCTION, BEQUEST FUNCTION

if risk_aver==1:
    u = lambda c: log(c)
    beq = lambda a: bequest_weight*np.log(a+bequest_luxury)
else:
    u = lambda c: (c**(1-risk_aver)-1)/(1-risk_aver)
    beq = lambda a: bequest_weight*((a+bequest_luxury)**(1-risk_aver)-1)/(1-risk_aver)    

u1 = lambda c: c**(-risk_aver)
u1inv = lambda u: u**(-1/risk_aver)

beq1 = lambda a: bequest_weight*(a+bequest_luxury)**(-risk_aver)

# INITIALIZE CONSUMPTION FUNCTION

conguess = np.zeros((nx,nyP,nb))
for ib in range(0,nb):
    for iy in range(0,nyP):
        conguess[:,iy,ib] = r*xgrid[:,iy]
        
# ITERATE ON EULER EQUATION WITH ENDOGENOUS GRID POINTS

con = conguess.copy()

Iter = 0
cdiff = 1000

while Iter <= max_iter and cdiff>tol_iter:
    Iter = Iter + 1
    sav = np.zeros((nx,nyP,nb))
    emuc = np.zeros((ns,nyP,nb))
    
    conlast = con.copy()
    muc  = u1(conlast)  # muc on grid for x'
    
    coninterp = np.empty((nyP,nb), dtype = object)
    mucinterp = np.empty((nyP,nb), dtype = object)
    
    ## create interpolating function for each yP'
    for ib in range(0,nb):
        for iy in range(0,nyP):
            if InterpMUC==1:
                mucinterp[iy,ib] = interp1d(xgrid[:,iy],muc[:,iy,ib],'linear',fill_value='extrapolate')
            elif InterpCon==1:
                coninterp[iy,ib] = interp1d(xgrid[:,iy],conlast[:,iy,ib],'linear',fill_value='extrapolate')
   
    ## loop over current persistent income and discount factor
    for ib in range(0,nb):
        for iy in range(0,nyP):
            
            muc1 = np.zeros((nx,nyP,nb))
            con1 = np.zeros((nx,nyP,nb))
            cash1 = np.zeros((nx,nyP,nb))

            ## loop over future income realizations and discount factor
            for ib2 in range(0,nb):
                for iyP2 in range(0,nyP):
                    for iyT2 in range(0,nyT):
                        if InterpMUC==1:
                            emuc[:,iy,ib] = emuc[:,iy,ib] + mucinterp[iyP2,ib2](R*sgrid + netlabincgrid[iyP2,iyT2])[:,0] * yPtrans[iy,iyP2] * yTdist[iyT2] * betatrans[ib,ib2]
                        elif InterpCon==1:
                            emuc[:,iy,ib] = emuc[:,iy,ib] + u1(coninterp[iyP2,ib2](R*sgrid + netlabincgrid[iyP2,iyT2])[:,0]) * yPtrans[iy,iyP2] * yTdist[iyT2] * betatrans[ib,ib2]
        
            muc1[:,iy,ib] = (1-dieprob)*betagrid[ib]*R*emuc[:,iy,ib]/(1+savtax*(sgrid[:,0]>=savtaxthresh)) + dieprob*beq1(sgrid[:,0])
            con1[:,iy,ib] = u1inv(muc1[:,iy,ib])
            cash1[:,iy,ib] = con1[:,iy,ib] + sgrid[:,0] + savtax*sgrid[:,0]*(sgrid[:,0]>=savtaxthresh)


            ## loop over current period cash on hand
            for ix in range(0,nx):
                if xgrid[ix,iy]<cash1[0,iy,ib]: # borrowing constraint binds
                    sav[ix,iy,ib] = borrow_lim
                else: # borrowing constraint does not bind
                    sav[ix,iy,ib] = lininterp1(cash1[:,iy,ib],sgrid,xgrid[ix,iy])

            con[:,iy,ib] = xgrid[:,iy] - sav[:,iy,ib]
            
    cdiff = np.max(abs(con-conlast))
    if Display>=1:
        print('Iteration no. ' + str(Iter) + ' max con fn diff is ' + str(cdiff))

# SIMULATE
if DoSimulate==1:
    
    yPindsim = np.zeros((Nsim,Tsim), dtype=int)
    logyTsim = np.zeros((Nsim,Tsim))
    logyPsim = np.zeros((Nsim,Tsim))
    ygrosssim = np.zeros((Nsim,Tsim))
    ynetsim = np.zeros((Nsim,Tsim))
    xsim = np.zeros((Nsim,Tsim))
    ssim = np.zeros((Nsim,Tsim))
    betaindsim = np.zeros((Nsim,Tsim), dtype=int)
    betasim = np.zeros((Nsim,Tsim))
    
    ## indicators for dying
    diesim = dierand<dieprob
        
    ## create interpolating functions
    savinterp = np.empty((nyP,nb), dtype = object)
    for ib in range(0,nb):
        for iy in range(0,nyP):
            savinterp[iy,ib] = interp1d(xgrid[:,iy],sav[:,iy,ib],'linear',fill_value='extrapolate')

    ## initialize permanent income
    it = 0
    yPindsim[yPrand[:,it]<= yPcumdist[0],it] = 0
    for iy in range(1,nyP):
        yPindsim[np.logical_and(yPrand[:,it]>yPcumdist[iy-1], yPrand[:,it]<=yPcumdist[iy]),it] = iy

    ## initialize discount factor
    it = 0
    betaindsim[betarand[:,it]<= betacumdist[0],it] = 0
    for ib in range(1,nb):
        betaindsim[np.logical_and(betarand[:,it]>betacumdist[ib-1], betarand[:,it]<=betacumdist[ib]),it] = ib

    ## loop over time periods
    for it in range(0,Tsim):
        if Display>=1 and (it+1)%100==0:
            print(' Simulating, time period ' + str(it+1))
        
        ## permanent income realization: note we vectorize simulations at once because
        ## of matlab, in other languages we would loop over individuals
        
        # people who dont die: use transition matrix
        if it>0:
            yPindsim[np.logical_and(diesim[:,it]==0, yPrand[:,it]<=yPcumtrans[yPindsim[:,it-1],0]),it] = 0
            for iy in range(1,nyP):
                yPindsim[np.logical_and(diesim[:,it]==0, np.logical_and(yPrand[:,it]>yPcumtrans[yPindsim[:,it-1],iy-1], yPrand[:,it]<=yPcumtrans[yPindsim[:,it-1],iy])),it] = iy
        
        ## people who die: descendendents draw income from stationary distribution
        yPindsim[np.logical_and(diesim[:,it]==1, yPrand[:,it]<=yPcumdist[0]),it] = 0
        for iy in range(1,nyP):
            yPindsim[np.logical_and(diesim[:,it]==1, np.logical_and(yPrand[:,it]>yPcumdist[iy-1], yPrand[:,it]<=yPcumdist[iy])),it] = iy

        logyPsim[:,it] = logyPgrid[yPindsim[:,it]][:,0]
        
        ## transitory income realization            
        if nyT>1:
            logyTsim[:,it] = -0.5*sd_logyT**2 + yTrand[:,it]*sd_logyT
        else:
            logyTsim[:,it] = 0

        ygrosssim[:,it] = np.exp(logyPsim[:,it] + logyTsim[:,it])
        ynetsim[:,it] = lumptransfer + (1-labtaxlow)*ygrosssim[:,it] - labtaxhigh*np.maximum(ygrosssim[:,it]-labtaxthresh,0)
        
        ## discount factor realization: note we vectorize simulations at once because
        ## of matlab, in other languages we would loop over individuals
        if it>1:
            betaindsim[betarand[:,it]<=betacumtrans[betaindsim[:,it-1],0],it] = 0
            for ib in range(1,nb):
                betaindsim[np.logical_and(betarand[:,it]>betacumtrans[betaindsim[:,it-1],ib-1], betarand[:,it]<=betacumtrans[betaindsim[:,it-1],ib]),it] = ib
        
        ## update cash on hand
        if it>0:
            xsim[:,it] = R*ssim[:,it-1] + ynetsim[:,it]
 
        ## savings choice
        for ib in range(0,nb):
            for iy in range(0,nyP):
                ssim[np.logical_and(yPindsim[:,it]==iy, betaindsim[:,it]==ib),it] = savinterp[iy,ib](xsim[np.logical_and(yPindsim[:,it]==iy, betaindsim[:,it]==ib),it])
        ssim[ssim<borrow_lim] = borrow_lim
                           
    ## convert to assets
    asim = (xsim - ynetsim)/R

    ## consumption
    csim = xsim - ssim - savtax*np.maximum(ssim-savtaxthresh,0)

# MAKE PLOTS
if MakePlots==1: 

    ## consumption policy function
    plt.plot(xgrid[:,0],con[:,0,0],'b-',label = 'Lowest income state')
    plt.plot(xgrid[:,nyP-1],con[:,nyP-1,0],'r-', label = 'Highest income state')
    plt.grid()
    plt.xlim((borrow_lim,xmax))
    plt.title('Consumption Policy Function')
    plt.legend()
    plt.show()

    ## savings policy function
    plt.plot(xgrid[:,0],sav[:,0,0]/xgrid[:,0],'b-')
    plt.plot(xgrid[:,nyP-1],sav[:,nyP-1,0]/xgrid[:,nyP-1],'r-')
    plt.plot(xgrid[:,0],np.zeros((nx,1)),'k',linewidth=0.5)
    plt.grid()
    plt.xlim((borrow_lim,xmax))
    plt.title('Savings Policy Function s/x')
    plt.show()
    
    ## nice zoom
    xlimits = (0,4)
    xlimind = np.ones((nx,nyP), dtype=bool)
    for iy in range(0,nyP):
        if np.min(xgrid[:,iy]) < xlimits[0]:
            xlimind[:,iy] = np.logical_and(xlimind[:,iy], (xgrid[:,iy]>=np.max(xgrid[xgrid[:,iy]<xlimits[0],iy])))
        elif np.min(xgrid[:,iy]) > xlimits[1]:
            xlimind[:,iy] = 0
        if np.max(xgrid[:,iy]) > xlimits[1]:
            xlimind[:,iy] = np.logical_and(xlimind[:,iy],(xgrid[:,iy]<=np.min(xgrid[xgrid[:,iy]>xlimits[1],iy])))
        elif np.max(xgrid[:,iy]) < xlimits[0]:
            xlimind[:,iy] = 0

    ## consumption policy function: zoomed in
    plt.plot(xgrid[xlimind[:,0],0],con[xlimind[:,0],0,0],'b-o',linewidth=2)
    plt.plot(xgrid[xlimind[:,nyP-1],nyP-1],con[xlimind[:,nyP-1],nyP-1,0],'r-o',linewidth=2)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Consumption: Zoomed')
    plt.show()

    ## savings policy function: zoomed in
    plt.plot(xgrid[xlimind[:,0],0],sav[xlimind[:,0],0,0]/xgrid[xlimind[:,0],0],'b-o',linewidth=2)
    plt.plot(xgrid[xlimind[:,nyP-1],nyP-1],sav[xlimind[:,nyP-1],nyP-1,0]/xgrid[xlimind[:,nyP-1],nyP-1],'r-o',linewidth=2)
    plt.plot(xgrid[:,0],np.zeros((nx,1)),'k',linewidth =0.5)
    plt.grid()
    plt.xlim(xlimits)
    plt.title('Savings (s/x): Zoomed')
    plt.show()
    
    ## income distribution
    plt.hist(ygrosssim[:,Tsim-1],50,facecolor=(0,0.5,0.5),edgecolor='blue')
    plt.ylabel('')
    plt.title('Income distribution')
    plt.show()

    ## asset distribution
    plt.hist(asim[:,Tsim-1],100,facecolor=(.7,.7,.7),edgecolor='black')
    plt.ylabel('')
    plt.title('Asset distribution')
    plt.show()

    ## convergence check
    plt.plot(range(1,Tsim+1),np.mean(asim,0),'k-',linewidth=1.5)
    plt.xlabel('Time Period')
    plt.title('Mean Asset Convergence')
    plt.show()

    ## asset distribution statistics
    aysim = asim[:,Tsim-1]/meangrosslabinc 
    print('Mean assets (relative to mean income): ' + str(np.mean(aysim)))
    print('Fraction borrowing constrained: ' + str(np.sum(aysim==borrow_lim)/Nsim * 100) + '%')
    print('10th Percentile: ' + str(np.quantile(aysim,.1)))
    print('50th Percentile: ' + str(np.quantile(aysim,.5)))
    print('90th Percentile: ' + str(np.quantile(aysim,.9)))
    print('99th Percentile: ' + str(np.quantile(aysim,.99)))
    
    ## table of parameters
    tabparam  = np.zeros((4,1))
    tabparam[0] = beta # discount factor
    tabparam[1] = risk_aver # risk aversion
    tabparam[2] = labtaxlow # labor tax rate
    tabparam[3] = lumptransfer/meangrosslabinc # relative to mean gross lab inc

    ## table of statistics
    tabstat = np.zeros((16,1))
    tabstat[0] = np.var(np.log(ygrosssim[:,Tsim-1])) # variance log gross earnings
    tabstat[1] = ginicoeff(ygrosssim[:,Tsim-1]) # gini gross earnings
    tabstat[2] = np.var(np.log(ynetsim[:,Tsim-1])) # variance log net earnings
    tabstat[3] = ginicoeff(ynetsim[:,Tsim-1]) # gini net earnings
    tabstat[4] = np.var(np.log(csim[:,Tsim-1])) # variance log consumption
    tabstat[5] = ginicoeff(csim[:,Tsim-1]) # gini consumption
    tabstat[6] = np.mean(aysim) # mean wealth (relative to mean earnings)
    tabstat[7] = np.quantile(aysim,.5) # median wealth (relative to mean earnings)
    tabstat[8] = ginicoeff(aysim) # wealth gini
    tabstat[9] = np.quantile(aysim,.9) / np.quantile(aysim,.5) # 90-10 wealth distribution
    tabstat[10] = np.quantile(aysim,.99) / np.quantile(aysim,.5) # 99-50 wealth distribution
    tabstat[11] = np.sum(aysim<=0)/Nsim # fraction  less than equal to zero
    tabstat[12] = np.sum(aysim<=0.05)/Nsim # fraction  less than equal to 5% av. annual gross earnings
    tabstat[13] = np.sum(aysim[aysim>=np.quantile(aysim,.9)]) / np.sum(aysim) # top 10% wealth share
    tabstat[14] = np.sum(aysim[aysim>=np.quantile(aysim,.99)]) / np.sum(aysim) # top 1% wealth share
    tabstat[15] = np.sum(aysim[aysim>=np.quantile(aysim,.999)]) / np.sum(aysim) # top 0.1% wealth share

    taboutput = np.vstack((tabparam, tabstat))

# COMPUTE MPCs
if ComputeMPC==1:
    
    ## theoretical mpc lower bound
    mpclim = R*((beta*R)**(-1/risk_aver))-1
        
    coninterp = list()
    mpc1 = np.zeros((nx,nyP))
    mpc2 = np.zeros((nx,nyP))
    
    for iy in range(0,nyP):
        ## create interpolating function
        coninterp.append(interp1d(xgrid[:,iy],con[:,iy].T,'linear',fill_value='extrapolate'))
        
        mpc1[:,iy] = ( coninterp[iy](xgrid[:,iy]+mpcamount1) - con[:,iy].T ) / mpcamount1
        mpc2[:,iy] = ( coninterp[iy](xgrid[:,iy]+mpcamount2) - con[:,iy].T ) / mpcamount2

    ## mpc functions
    plt.plot(xgrid,mpc1[:,int((nyP+1)/2)],'b-',linewidth=1,label='Amount 1')
    plt.plot(xgrid,mpc2[:,int((nyP+1)/2)],'b--',linewidth=1,label='Amount 2')
    plt.plot(xgrid[:,0],mpclim*np.ones(np.shape(xgrid[:,0])),'k:',linewidth=2,label='Theoretical MPC limit = '+str(mpclim))
    plt.grid()
    plt.xlim((0,10))
    plt.title('MPC Function: median persistant income state')
    mpc_handles, mpc_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(mpc_labels, mpc_handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    ## mpc distribution
    mpc1sim = np.zeros((Nsim,1))
    mpc2sim = np.zeros((Nsim,1))
    for iy in range(0,nyP):
        mpc1sim[yPindsim[:,Tsim-1]==iy,0] = ( coninterp[iy](xsim[yPindsim[:,Tsim-1]==iy,Tsim-1]+mpcamount1) - coninterp[iy](xsim[yPindsim[:,Tsim-1]==iy,Tsim-1]) ) / mpcamount1
        mpc2sim[yPindsim[:,Tsim-1]==iy,0] = ( coninterp[iy](xsim[yPindsim[:,Tsim-1]==iy,Tsim-1]+mpcamount2) - coninterp[iy](xsim[yPindsim[:,Tsim-1]==iy,Tsim-1]) ) / mpcamount2
    
    plt.hist(mpc1sim,np.linspace(0,1.5,76),facecolor=(.7,.7,.7),edgecolor='black',linestyle='-')
    plt.grid()
    plt.ylabel('')
    plt.title('MPC distribution')
    
    # mpc distribution statistics
    print('Mean MPC amount 1: ' + str(np.mean(mpc1sim)))
    print('Mean MPC amount 2: ' + str(np.mean(mpc2sim)))
    plt.show()