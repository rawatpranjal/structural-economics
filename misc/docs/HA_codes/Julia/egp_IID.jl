# Endogenous Grid Points with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

using Random, Interpolations, NLsolve, Plots
include("discrete_normal.jl")
include("lininterp1.jl")

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
mpcamount1 = 1.0e-10 # approximate thoeretical mpc
mpcamount2 = 0.10 # one percent of average income: approx $500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1
ComputeMPC = 1

# DRAW RANDOM NUMBERS
Random.seed!(2020)
yrand = rand(Nsim,Tsim)

# SET UP GRIDS

## assets
agrid = range(0,1,length=na)
agrid = agrid.^(1 ./ agrid_par)
agrid = borrow_lim .+ (amax.-borrow_lim).*agrid

## income: disretize normal distribution
width = nlsolve(x -> discrete_normal(ny,mu_y,sd_y,x...)[1],[2.0]).zero
temp, ygrid, ydist = discrete_normal(ny,mu_y,sd_y,width...)
ycumdist = cumsum(ydist)

# UTILITY FUNCTION

u(c) = (c.^(1-risk_aver)-1)./(1-risk_aver)
u1(c) = c.^(-risk_aver)
u1inv(u) = u.^(-1 ./risk_aver)

# INITIALIZE CONSUMPTION FUNCTION

conguess = zeros(na,ny)
for iy = 1:ny
    conguess[:,iy] = r.*agrid.+ygrid[iy]
end

# ITERATE ON EULER EQUATION WITH ENDOGENOUS GRID POINTS

con = copy(conguess)

iter = 0
cdiff = 1000

while iter <= max_iter && cdiff>tol_iter
    iter = iter + 1
    global sav = zeros(na,ny)
    
    conlast = copy(con)
    
    emuc = u1(conlast)*ydist
    muc1 = beta.*R.*emuc
    con1 = u1inv(muc1)
   
    ## loop over income
    ass1 = zeros(na,ny)
    for iy = 1:ny
        
        ass1[:,iy] = (con1 .+ agrid .-ygrid[iy])./R
        
        ## loop over current period ssets
        for ia  = 1:na 
            if agrid[ia]<ass1[1,iy] # borrowing constraint binds
                sav[ia,iy] = borrow_lim                
            else # borrowing constraint does not bind;
                sav[ia,iy] = lininterp1(ass1[:,iy],agrid,agrid[ia])
            end                
        end
        con[:,iy] = R.*agrid .+ ygrid[iy] - sav[:,iy]
    end

    cdiff = maximum(abs.(con - conlast))
    if Display>=1
        println("Iteration no. " * string(iter), " max con fn diff is " * string(cdiff))
    end
end

# SIMULATE
if DoSimulate==1

    yindsim = zeros(Int,Nsim,Tsim)
    asim = zeros(Nsim,Tsim)
    
    ## create interpolating function
    savinterp = Array{Any}(undef,ny)
    for iy = 1:ny
        savinterp[iy] = interpolate((agrid,), sav[:,iy], Gridded(Linear()))
    end
    
    ## loop over time periods
    for it = 1:Tsim
        if Display>=1 && mod(it,100)==0
            println("Simulating, time period " * string(it))
        end
        
        ## income realization: note we vectorize simulations at once because
        ## of matlab, in other languages we would loop over individuals
        yindsim[yrand[:,it].<=ycumdist[1],it] .= 1
        for iy = 2:ny
            yindsim[(yrand[:,it].> ycumdist[iy-1]) .& (yrand[:,it].<=ycumdist[iy]),it] .= iy
        end
        
        ## asset choice
        if it<Tsim
            for iy = 1:ny
                asim[yindsim[:,it].==iy,it+1] = savinterp[iy](asim[yindsim[:,it].==iy,it])
            end
        end
    end
    
    ## assign actual income values
    ysim = ygrid[yindsim]

end

# MAKE PLOTS
if MakePlots==1
    
    ## consumption policy function
    p1 = plot(agrid, [con[:,1] con[:,ny]], xlims=(0,amax), title="Consumption", color=[:blue :red], label=["Lowest income state" "Highest income state"])
    display(p1)
    
    ## savings policy function
    p2 = plot(agrid, [sav[:,1].-agrid[:,1] sav[:,ny].-agrid[:,1]], xlims=(0,amax), title="Savings", color=[:blue :red], legend=false)
    plot!(agrid, zeros(na,1), color=:black, lw=0.5)
    display(p2)
    
    ## nice zoom
    xlimits = (0,1)
    xlimind = trues(na)
    if minimum(agrid) < xlimits[1]
        xlimind = xlimind .& (agrid.>=maximum(agrid[agrid<xlimits[1]]))
    elseif minimum(agrid) > xlimits[2]
        xlimind .= 0
    end
    if maximum(agrid) > xlimits[2]
        xlimind = xlimind .& (agrid.<=minimum(agrid[agrid.>xlimits[2]]))
    elseif maximum(agrid) < xlimits[1]
        xlimind .= 0
    end

    ## consumption policy function: zoomed in
    p3 = plot(agrid[xlimind], [con[xlimind,1] con[xlimind,ny]], xlims=xlimits, title="Consumption: Zoomed", marker=:circle, color=[:blue :red], linewidth=2, legend=false)
    display(p3)

    ## savings policy function: zoomed in
    p4 = plot(agrid[xlimind], [sav[xlimind,1].-agrid[xlimind] sav[xlimind,ny].-agrid[xlimind]], xlims=xlimits, title="Savings: Zoomed (a'-a)", marker=:circle, color=[:blue :red], linewidth=2, legend=false)
    plot!(agrid, zeros(na,1), color=:black, lw=0.5)
    display(p4)

    ## income distribution
    p5 = histogram(ysim[:,Tsim], bins=[2*ygrid[1]-ygrid[2];ygrid].+(ygrid[2]-ygrid[1])/2, title="Income distribution", color=RGB(0,0.5,0.5), linecolor=:blue, legend=false)
    display(p5)

    ## asset distribution
    p6 = histogram(asim[:,Tsim], nbins=100, title="Asset distribution", color=RGB(.7,.7,.7), linecolor=:black, legend=false)
    display(p6)

    ## convergence check
    p7 = plot(1:Tsim, mean(asim,dims=1)', title="Mean Asset Convergence", xlabel="Time Period", color=:black, lw=1.5, legend=false)
    display(p7)

    ## asset distribution statistics
    aysim = asim[:,Tsim]./mean(ysim[:,Tsim])
    println("Mean assets (relative to mean income): " * string(mean(aysim)))
    println("Fraction borrowing constrained: " * string(sum(aysim.==borrow_lim)/Nsim * 100) * '%')
    println("10th Percentile: " * string(quantile(aysim,.1)))
    println("50th Percentile: " * string(quantile(aysim,.5)))
    println("90th Percentile: " * string(quantile(aysim,.9)))
    println("99th Percentile: " * string(quantile(aysim,.99)))
end

# COMPUTE MPCs
if ComputeMPC==1
    
    ## theoretical mpc lower bound
    mpclim = R*((beta*R)^(-1/risk_aver))-1
    
    coninterp = Array{Any}(undef,ny)
    mpc1 = zeros(na,ny)
    mpc2 = zeros(na,ny)
    
    for iy = 1:ny
        ## create interpolating function
        coninterp[iy] = extrapolate(interpolate((agrid,), con[:,iy], Gridded(Linear())), Line())

        mpc1[:,iy] = ( coninterp[iy](agrid.+mpcamount1) - con[:,iy] ) ./ mpcamount1
        mpc2[:,iy] = ( coninterp[iy](agrid.+mpcamount2) - con[:,iy] ) ./ mpcamount2
        
    end
    
    ## mpc functions
    p8 = plot([agrid agrid],[mpc1[:,1] mpc2[:,1]], xlims=(0,10), linestyle=[:solid :dash], color=:blue, label=["Lowest income state: amount 1" "Lowest income state: amount 2"])
    plot!([agrid agrid],[mpc1[:,ny] mpc2[:,ny]], linestyle=[:solid :dash], color=:red, label=["Highest income state: amount 1" "Highest income state: amount 2"])
    plot!(agrid,mpclim.*ones(size(agrid)),color=:black,linestyle=:dot,linewidth=2,label="Theoretical MPC limit = " * string(mpclim))
    display(p8)
    
    ## mpc distribution
    mpc1sim = zeros(Nsim,1)
    mpc2sim = zeros(Nsim,1)
    for iy = 1:ny
        mpc1sim[yindsim[:,Tsim].==iy] = ( coninterp[iy](asim[yindsim[:,Tsim].==iy,Tsim].+mpcamount1) - coninterp[iy](asim[yindsim[:,Tsim].==iy,Tsim]) ) ./ mpcamount1
        mpc2sim[yindsim[:,Tsim].==iy] = ( coninterp[iy](asim[yindsim[:,Tsim].==iy,Tsim].+mpcamount2) - coninterp[iy](asim[yindsim[:,Tsim].==iy,Tsim]) ) ./ mpcamount2
    end
    
    p9 = histogram(mpc1sim,bins=0:0.02:1.5, title="MPC distribution",color=RGB(.7,.7,.7),legend=false)
    display(p9)
        
    # mpc distribution statistics
    println("Mean MPC amount 1: " * string(mean(mpc1sim)))
    println("Mean MPC amount 2: " * string(mean(mpc2sim)))
end