using Distributions
function discrete_normal(n,mu,sigma,width)

# creates equally spaced approximation to normal distribution
# n is number of points
# mu is mean
# sigma is standard deviation
# width is the multiple of stand deviation for the width of the grid
# f is the error in the approximation
# x gives the location of the points
# p is probabilities

    x = collect(range(mu-width*sigma,mu+width*sigma,length=n))
    if n==2
        p = 0.5.*ones(n)
    elseif n>2    
        p  = zeros(n)
        p[1] = cdf(Normal(mu,sigma), x[1] + 0.5*(x[2]-x[1]))
        for i = 2:n-1
            p[i] = cdf(Normal(mu,sigma), x[i] + 0.5*(x[i+1]-x[i])) - cdf(Normal(mu,sigma), x[i] - 0.5*(x[i]-x[i-1]))
        end
        p[n] = 1 - sum(p[1:n-1])
    end

    Ex = x'*p
    SDx = sqrt((x.^2)'*p .- Ex.^2)

    f = SDx.-sigma
            
    return f,x,p
            
end