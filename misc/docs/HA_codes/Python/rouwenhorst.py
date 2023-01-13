import numpy as np
def rouwenhorst(n, mu, sigma, rho):
    
    # grid
    width = np.sqrt((n-1) * sigma**2 / ( 1 - rho**2))
    grid = np.linspace( mu-width, mu + width, n).reshape(n,1)
    
    # transition matrix
    p0 = (1 + rho) / 2
    trans = np.array([[p0, 1-p0], [1-p0, p0]])

    if n > 2:
        for i in range(1,10):
            cstr_temp = np.zeros((len(trans[:,0]), 1))
            trans = p0*np.block([[trans, cstr_temp], [cstr_temp.T, 0]]) + (1-p0)*np.block([[cstr_temp, trans], [cstr_temp.T, 0]]) + (1-p0)*np.block([[cstr_temp.T, 0], [trans, cstr_temp]]) + p0*np.block([ [cstr_temp.T, 0], [cstr_temp, trans]])
        trans = np.apply_along_axis(lambda x: x/np.sum(trans,axis=1), 0, trans)
    
    ## ergodic distribution
    dist = np.ones((1,n))/n
    for i in range(1,101):
        dist = dist @ np.linalg.matrix_power(trans,i)
  
    return grid, trans, dist.T