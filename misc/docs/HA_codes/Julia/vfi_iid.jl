# Value Function Iteration with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

using Random, NLsolve, Plots
include("discrete_normal.jl")

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
na = 500
amax = 20
borrow_lim = 0
agrid_par = 1 # 1 for linear, 0 for L-shaped

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 50000
Tsim = 500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1

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

if risk_aver==1
    u(c) = log.(c)
else
    u(c) = (c.^(1-risk_aver).-1)./(1-risk_aver)
end

# INITIALIZE VALUE FUNCTION

Vguess = zeros(na,ny)
for iy = 1:ny
    Vguess[:,iy] .= u(r.*agrid[1]+ygrid[iy])./(1-beta)
end

### Vguess = ones(na,ny)

# ITERATE ON VALUE FUNCTION

V = copy(Vguess)

Vdiff = 1
iter = 0

while iter <= max_iter && Vdiff > tol_iter
    iter = iter + 1
    Vlast = copy(V)
    V = zeros(na,ny)
    global sav = zeros(na,ny)
    global savind = zeros(Int,na,ny)
    global con = zeros(na,ny)

    ## loop over assets
    for ia = 1:na
        
        ## loop over income
        for iy = 1:ny
            cash = R.*agrid[ia] + ygrid[iy]
            Vchoice = u(max.(cash.-agrid,1.0e-10)) .+ beta.*(Vlast*ydist)           
            V[ia,iy] = maximum(Vchoice)
            savind[ia,iy] = argmax(Vchoice)[1]
            sav[ia,iy] = agrid[savind[ia,iy]]
            con[ia,iy] = cash .- sav[ia,iy]
        end
    end
    
    Vdiff = maximum(abs.(V-Vlast))
    if Display >= 1
        println("Iteration no. " * string(iter), " max val fn diff is " * string(Vdiff))
    end
end

# SIMULATE
if DoSimulate == 1
    yindsim = zeros(Int,Nsim,Tsim)
    aindsim = zeros(Int,Nsim,Tsim)
    
    ## initial assets
    aindsim[:,1] .= 1
    
    ## loop over time periods
    for it = 1:Tsim
        if Display >= 1 && mod(it,100)==0
            println(" Simulating, time period " * string(it))
        end
        
        ### income realization: note we vectorize simulations at once because
        ### of matlab, in other languages we would loop over individuals
        yindsim[yrand[:,it].<=ycumdist[1],it] .= 1
        for iy = 2:ny
            yindsim[(yrand[:,it].>ycumdist[iy-1]) .& (yrand[:,it].<=ycumdist[iy]),it] .= iy
        end
        
        ## asset choice
        if it < Tsim
            for iy = 1:ny
                aindsim[yindsim[:,it].==iy,it+1] = savind[aindsim[yindsim[:,it].==iy,it],iy]
            end
        end
    end

    ## assign actual asset and income values
    asim = agrid[aindsim]
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
    plot!(show=true)
    display(p3)

    ## savings policy function: zoomed in
    p4 = plot(agrid[xlimind], [sav[xlimind,1].-agrid[xlimind] sav[xlimind,ny].-agrid[xlimind]], xlims=xlimits, title="Savings: Zoomed (a'-a)", marker=:circle, color=[:blue :red], linewidth=2, legend=false)
    plot!(agrid, zeros(na,1), color=:black, lw=0.5)
    display(p4)

    ## income distribution
    p5 = histogram(ysim[:,Tsim], bins=[2*ygrid[1]-ygrid[2];ygrid].+(ygrid[2]-ygrid[1])/2, title="Income distribution", color=RGB(0,0.5,0.5), linecolor=:blue, legend=false)
    display(p5)

    ## asset distribution
    p6 = histogram(asim[:,Tsim], bins=-0.025:0.05:1.975, title="Asset distribution", color=RGB(.7,.7,.7), linecolor=:black, legend=false)
    display(p6)

    ## convergence check
    p7 = plot(1:Tsim, mean(asim,dims=1)', title="Mean Asset Convergence", xlabel="Time Period", color=:black, lw=1.5, legend=false)
    display(p7)

    ## asset distribution statistics
    aysim = asim[:,Tsim]./mean(ysim[:,Tsim])
    println("Mean assets: " * string(mean(aysim)))
    println("Fraction borrowing constrained: " * string(sum(aysim.==borrow_lim)/Nsim * 100) * '%')
    println("10th Percentile: " * string(quantile(aysim,.1)))
    println("50th Percentile: " * string(quantile(aysim,.5)))
    println("90th Percentile: " * string(quantile(aysim,.9)))
    println("99th Percentile: " * string(quantile(aysim,.99)))
end