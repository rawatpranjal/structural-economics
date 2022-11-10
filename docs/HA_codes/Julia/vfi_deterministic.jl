# Deterministic Value Function Iteration
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

using Random, Interpolations, Plots

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

# DRAW RANDOM NUMBERS
Random.seed!(2020)
arand = rand(Nsim)

# SET UP GRIDS

## assets
agrid = range(0,1,length=na)
agrid = agrid.^(1 ./ agrid_par)
agrid = borrow_lim .+ (amax.-borrow_lim).*agrid

# UTILITY FUNCTION

if risk_aver==1
    u(c) = log.(c)
else
    u(c) = (c.^(1-risk_aver).-1)./(1-risk_aver)
end

u1(c) = c.^(-risk_aver)

# INITIALIZE VALUE FUNCTION

Vguess = u(r.*agrid.+y)./(1-beta)

# ITERATE ON VALUE FUNCTION

V = copy(Vguess)

Vdiff = 1
iter = 0

while iter <= max_iter && Vdiff > tol_iter
    iter = iter + 1
    Vlast = copy(V)
    V = zeros(na)
    global sav = zeros(na)
    global savind = zeros(Int,na)
    global con = zeros(na)

    ## loop over assets
    for ia = 1:na
        
        cash = R.*agrid[ia] + y
        Vchoice = u(max.(cash.-agrid,1.0e-10)) + beta.*Vlast         
        V[ia] = maximum(Vchoice)
        savind[ia] = argmax(Vchoice)[1]
        sav[ia] = agrid[savind[ia]]
        con[ia] = cash .- sav[ia]
    end
    
    Vdiff = maximum(abs.(V-Vlast))
    if Display>=1
        println("Iteration no. " * string(iter), " max val fn diff is " * string(Vdiff))
    end
end

# SIMULATE
if DoSimulate==1
    yindsim = zeros(Int,Nsim,Tsim)
    aindsim = zeros(Int,Nsim,Tsim)
    
    ## initial assets: uniform on [borrow_lim, amax]    
    ainitial = borrow_lim .+ arand.*(amax-borrow_lim)
    
    ## allocate to nearest point on agrid;
    aindsim[:,1] = interpolate((agrid,), 1:na, Gridded(Constant())).(ainitial)
    
    ## loop over time periods
    for it = 1:Tsim
        if Display >= 1 && mod(it,100)==0
            println(" Simulating, time period " * string(it))
        end
        ## asset choice
        if it<Tsim
            aindsim[:,it+1] = savind[aindsim[:,it]]
        end
    end
    
    ## assign actual asset and income values;
    asim = agrid[aindsim]
    csim = R.*asim[:,1:Tsim-1] .+ y .- asim[:,2:Tsim]
end

# MAKE PLOTS
if MakePlots==1
    
    ## consumption policy function
    p1 = plot(agrid, con, xlims=(0,amax), title="Consumption", color=:blue, legend=false)
    display(p1)
    
    ## savings policy function
    p2 = plot(agrid, sav.-agrid, xlims=(0,amax), title="Savings", color=:blue, legend=false)
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
    p3 = plot(agrid[xlimind], con[xlimind], xlims=xlimits, title="Consumption: Zoomed", marker=:circle, color=:blue, linewidth=2, legend=false)
    plot!(show=true)
    display(p3)

    ## savings policy function: zoomed in
    p4 = plot(agrid[xlimind], sav[xlimind].-agrid[xlimind], xlims=xlimits, title="Savings: Zoomed (a'-a)", marker=:circle, color=:blue, linewidth=2, legend=false)
    plot!(agrid, zeros(na), color=:black, lw=0.5)
    display(p4)
       
    ## asset dynamics distribution
    p5 = plot(repeat((1:Tsim),1,Nsim), asim', title="Asset Dynamics", legend=false)
    display(p5)
    
    ## consumption dynamics distribution
    p6 = plot(repeat((1:Tsim-1),1,Nsim), csim', title="Consumption Dynamics", legend=false)
    display(p6)

end