# Aiyagari model
# Endogenous Grid Points with IID Income
# Greg Kaplan 2017
# Translated by Tom Sweeney Jan 2021

using Random, Interpolations, NLsolve, Plots
include("discrete_normal.jl")
include("lininterp1.jl")

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
KLratioguess = ((rguess + deprec)/capshare)^(1/(capshare-1))

# OPTIONS
Display = 1
MakePlots = 1

## which function to interpolation 
InterpCon = 0
InterpEMUC = 1

## tolerance for non-linear solver
TolX=1.0e-6

# UTILITY FUNCTION

if risk_aver==1
    u(c) = log.(c)
else
    u(c) = (c.^(1-risk_aver).-1)./(1-risk_aver)
end 

u1(c) = c.^(-risk_aver)
u1inv(u) = u.^(-1 ./risk_aver)

# SET UP GRIDS

## assets
agrid = range(0,1,length=na)
agrid = agrid.^(1 ./ agrid_par)
agrid = borrow_lim .+ (amax.-borrow_lim).*agrid

## income: disretize normal distribution
width = nlsolve(x -> discrete_normal(ny,mu_y,sd_y,x...)[1],[2.0]).zero
temp, ygrid, ydist = discrete_normal(ny,mu_y,sd_y,width...)
ycumdist = cumsum(ydist)

# DRAW RANDOM NUMBERS
Random.seed!(2021)
yrand = rand(Nsim,Tsim)

# SIMULATE LABOR EFFICIENCY REALIZATIONS
if Display>=1
    println("Simulating labor efficiency realizations in advance")
end
yindsim = zeros(Int,Nsim,Tsim)
    
for it = 1:Tsim

    # income realization: note we vectorize simulations at once because
    # of matlab, in other languages we would loop over individuals
    yindsim[yrand[:,it].<=ycumdist[1],it] .= 1
    for iy = 2:ny
        yindsim[(yrand[:,it].>ycumdist[iy-1]) .& (yrand[:,it].<=ycumdist[iy]),it] .= iy;
    end
end
    
ysim = ygrid[yindsim]

# ITERATE OVER KL RATIO
KLratio = KLratioguess

iterKL = 0
KLdiff = 1

while iterKL<=maxiter_KL && abs(KLdiff)>tol_KL
    iterKL = iterKL + 1

    r = capshare.*KLratio^(capshare-1) - deprec
    R = 1+r
    wage = (1-capshare).* KLratio^capshare

    ## rescale efficiency units of labor so that output = 1
    yscale = (KLratio^(-capshare))./(ygrid'*ydist)
    
    ## initialize consumption function in first iteration only
    if iterKL==1
        conguess = zeros(na,ny)
        for iy = 1:ny
            conguess[:,iy] = r.*agrid .+ wage.* yscale.*ygrid[iy]
        end
        global con = copy(conguess)
     end

    ## solve for policy functions with EGP
    
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

            ass1[:,iy] = (con1 .+ agrid .-wage.* yscale.*ygrid[iy])./R

            ## loop over current period ssets
            for ia  = 1:na 
                if agrid[ia]<ass1[1,iy] # borrowing constraint binds
                    sav[ia,iy] = borrow_lim                
                else # borrowing constraint does not bind;
                    sav[ia,iy] = lininterp1(ass1[:,iy],agrid,agrid[ia])
                end                
            end
            con[:,iy] = R.*agrid .+ wage.*yscale.*ygrid[iy] - sav[:,iy]
        end
     
        cdiff = maximum(abs.(con - conlast))
        if Display>=2
            println("Iteration no. " * string(iter), " max con fn diff is " * string(cdiff))
        end
    end    


    ## simulate: start at assets from last interation
    if iterKL==1
        global asim = zeros(Nsim,Tsim)
    elseif iterKL>1
        # asim[:,1] = Ea.*ones(Nsim,1)
        asim[:,1] = asim[:,Tsim]
    end
    
    ## create interpolating function
    savinterp = Array{Any}(undef,ny)
    for iy = 1:ny
        savinterp[iy] = interpolate((agrid,), sav[:,iy], Gridded(Linear()))
    end
    
    ## loop over time periods
    for it = 1:Tsim
        if Display>=2 && mod(it,100)==0
            println("Simulating, time period " * string(it))
        end
        
        ## asset choice
        if it<Tsim
            for iy = 1:ny
                asim[yindsim[:,it].==iy,it+1] = savinterp[iy](asim[yindsim[:,it].==iy,it])
            end
        end
    end

    ## assign actual labor income values
    labincsim = wage.*yscale.*ysim

    ## mean assets and efficiency units
    global Ea = mean(asim[:,Tsim])
    L = yscale.*mean(ysim[:,Tsim])
    
    KLrationew = Ea./ L
    
    KLdiff = KLrationew./KLratio - 1
    if Display>=1
        println("Equm iter " * string(iterKL) * ", r = " * string(r) * ", KL ratio: " * string(KLrationew) * " KL diff: " * string(KLdiff*100) * "%")
    end

    KLratio = (1-step_KL)*KLratio + step_KL*KLrationew
end

# MAKE PLOTS
if MakePlots==1
    
    ## consumption policy function
    p1 = plot(agrid, [con[:,1] con[:,ny]], xlims=(0,amax), title="Consumption Policy Function", color=[:blue :red], label=["Lowest income state" "Highest income state"])
    display(p1)
    
    ## savings policy function
    p2 = plot(agrid, [sav[:,1].-agrid[:,1] sav[:,ny].-agrid[:,1]], xlims=(0,amax), title="Savings Policy Function", color=[:blue :red], legend=false)
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
    p6 = histogram(asim[:,Tsim], nbins=100, title="Asset distribution", color=RGB(.7,.7,.7), linecolor=:black, legend=false)
    display(p6)
    
    ## convergence check
    p7 = plot(1:Tsim, mean(asim,dims=1)', ylims=(0,2*Ea), title="Mean Asset Convergence", xlabel="Time Period", color=:black, lw=1.5, legend=false)
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