# Euler Equation Iteration with IID Income
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
na = 30
amax = 30
borrow_lim = 0
agrid_par = 0.4 # 1 for linear, 0 for L-shaped

## computation
max_iter = 1000
tol_iter = 1.0e-6
Nsim = 50000
Tsim = 500

# OPTIONS
Display = 1
DoSimulate = 1
MakePlots = 1

## which function to interpolation 
InterpCon = 1
InterpEMUC = 0

## tolerance for non-linear solver
TolX = 1.0e-6

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
ycumdist = cumsum(ydist,dims=1)

# UTILITY FUNCTION

if risk_aver==1
    u(c) = log.(c)
else
    u(c) = (c.^(1-risk_aver).-1)./(1-risk_aver)
end

u1(c) = c.^(-risk_aver)

# INITIALIZE CONSUMPTION FUNCTION

conguess = zeros(na,ny)
for iy = 1:ny
    conguess[:,iy] = r.*agrid.+ygrid[iy]
end

# ITERATE ON EULER EQUATION

con = copy(conguess)
emuc = u1(con)*ydist

iter = 0
cdiff = 1000

if InterpCon==1
    fn_eeqn_c(a) = u1(cash-a)-beta.*R.*(u1([lininterp1(agrid,conlast[:,iy],a) for iy=1:ny])'*ydist)[1]
end

while iter<=max_iter && cdiff>tol_iter
    iter = iter + 1
    global conlast = copy(con)
    global sav = zeros(na,ny)
    
    ## loop over assets
    for ia = 1:na
        
        ## loop over income
        for iy = 1:ny
            global cash = R.*agrid[ia] + ygrid[iy]
            
            ## use consumption interpolation
            if InterpCon==1
                if fn_eeqn_c(borrow_lim)>=0 # check if borrowing constrained
                    sav[ia,iy] = borrow_lim
                else
                    sav[ia,iy] = nlsolve(x -> fn_eeqn_c(x...),[cash-conlast[ia,iy]],xtol=TolX).zero[1]
                end    
                
            ## use expected marginal utility interpolation
            elseif InterpEMUC==1
                if u1(cash-borrow_lim) >= beta.*R.*lininterp1(agrid,emuc,borrow_lim) # check if borrowing constrained
                    sav[ia,iy] = borrow_lim
                else
                    sav[ia,iy] = nlsolve(x -> u1(cash.-x).-beta.*R.*lininterp1(agrid,emuc,x...),[cash-conlast[ia,iy]],xtol=TolX).zero[1]
                end
                
            end    
            con[ia,iy] = cash .- sav[ia,iy]
        end
    end
    
    emuc = u1(con)*ydist
    
    cdiff = maximum(abs.(con-conlast))
    if Display >= 1
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
        if Display >=1 && mod(it,100)==0
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
    println("Mean assets (relative to mean income): " * string(mean(aysim)))
    println("Fraction borrowing constrained: " * string(sum(aysim.==borrow_lim)/Nsim * 100) * '%')
    println("10th Percentile: " * string(quantile(aysim,.1)))
    println("50th Percentile: " * string(quantile(aysim,.5)))
    println("90th Percentile: " * string(quantile(aysim,.9)))
    println("99th Percentile: " * string(quantile(aysim,.99)))
end