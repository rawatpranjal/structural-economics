def discrete_normal(n,mu,sigma,width):

# creates equally spaced approximation to normal distribution
# n is number of points
# mu is mean
# sigma is standard deviation
# width is the multiple of stand deviation for the width of the grid
# f is the error in the approximation
# x gives the location of the points
# p is probabilities

    import numpy as np
    from scipy.stats import norm

    x = np.linspace(mu-width*sigma,mu+width*sigma,n).reshape(n,1)
    if n==2:
        p = 0.5*np.ones(n).reshape(n,1)
    elif n>2:
        p  = np.zeros(n).reshape(n,1)
        p[0] = norm.cdf(x[0] + 0.5*(x[1]-x[0]),mu,sigma)
        for i in range(1,n-1):
            p[i] = norm.cdf(x[i] + 0.5*(x[i+1]-x[i]),mu,sigma) - norm.cdf(x[i] - 0.5*(x[i]-x[i-1]),mu,sigma)
        p[n-1] = 1 - sum(p[0:n-1])
    
    Ex = x.T @ p
    SDx = np.sqrt((x.T**2) @ p - Ex**2)

    f = (SDx-sigma)[0,0]
            
    return f,x,p