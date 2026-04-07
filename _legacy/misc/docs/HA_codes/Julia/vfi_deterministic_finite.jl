# Deterministic Value Function Iteration
# Greg Kaplan 2017
# Translated by Tom Sweeney Dec 2020

using Random, Interpolations, Plots

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
y = zeros(T)
y[1:30] = 1:30
y[31:50] .= 5

## rescale income so that average income = 1
y = y./sum(y)

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
agrid = range(0,1,length=na)
agrid = agrid.^(1 ./ agrid_par)
agrid = borrow_lim .+ (amax.-borrow_lim).*agrid

## put explicit point at a=0
agrid[agrid.==minimum(abs.(agrid.-0))] .= 0

# UTILITY FUNCTION

if risk_aver==1
    u(c) = log.(c)
else
    u(c) = (c.^(1-risk_aver).-1)./(1-risk_aver)
end

u1(c) = c.^(-risk_aver)

# INITIALIZE ARRAYS
V = zeros(na,T)
con = zeros(na,T)
sav = zeros(na,T)
savind = zeros(Int,na,T)

# DECISIONS AT t=T
savind[:,T] .= findfirst(x->x==0, agrid)
sav[:,T] .= 0
con[:,T]= R.*agrid .+ y[T] .- sav[:,T]
V[:,T] = u(con[:,T])

# SOlVE VALUE FUNCTION BACKWARD 

for it = T-1:-1:1
    if Display>=1 
        println("Solving at age: " * string(it))
    end
    
    ## loop over assets
    for ia = 1:na
             
        cash = R.*agrid[ia] + y[it]
        Vchoice = u(max.(cash.-agrid,1.0e-10)) + beta.*V[:,it+1]  
        V[ia,it] = maximum(Vchoice)
        savind[ia,it] = argmax(Vchoice)[1]
        sav[ia,it] = agrid[savind[ia,it]]
        con[ia,it] = cash .- sav[ia,it]
    end
    
end

# SIMULATE
if DoSimulate==1
    aindsim = zeros(Int,T+1)
    
    ## initial assets: uniform on [borrow_lim, amax]    
    ainitial = 0
    
    ## allocate to nearest point on agrid;
    aindsim[1] = interpolate((agrid,), 1:na, Gridded(Constant())).(ainitial)
    
    ## loop over time periods
    for it = 1:T
        println(" Simulating, time period " * string(it))
        
        ## asset choice
        aindsim[it+1] = savind[aindsim[it],it]
    end
    
    ## assign actual asset and income values;
    asim = agrid[aindsim]
    csim = R.*asim[1:T] .+ y .- asim[2:T+1]
end

# MAKE PLOTS
if MakePlots==1 
    
    ## consumption and income path
    p1 = plot([1:50 1:50],[y csim],color=[:black :red],linestyle=[:solid :dash],labels=["Income" "Consumption"],title="Income and Consumption")
    display(p1)

    ## wealth path function
    p2 = plot(0:50,asim,color=:blue,title="Wealth",legend=false)
    plot!(agrid,zeros(na),color=:black,linewidth=0.5)
    display(p2)        

end