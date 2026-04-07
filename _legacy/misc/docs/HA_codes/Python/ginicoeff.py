import numpy as np
import warnings

def ginicoeff(In,dim=0,nosamplecorr=False):
    
    if np.ndim(In) == 0:
        In = np.array([In], dtype = float)
    else:
        In = np.array(In, dtype = float)
    
    # Negative values or one-element series (not admitted)
    IDXnan = np.isnan(In)
    IDX = np.logical_or(np.any(In < 0,dim), np.sum(np.logical_not(IDXnan),dim)<2)
    if dim == 0:
        In[:,IDX] = 0
    else:
        In[IDX,:] = 0

    if np.any(IDX):
        warnings.warn('Check IDX for negative values or one-element series, IDX = ' + str(IDX))

    # Total numel
    totNum = np.sum(np.logical_not(IDXnan),dim)

    # Replace NaNs
    In[IDXnan] = 0

    # Sort In
    In = np.sort(In,dim)

    # Calculate frequencies for each series
    freq = np.flip(np.cumsum(np.ones(np.shape(In)),dim),dim)

    # Totals
    tot = np.array(np.sum(In,dim))
    if np.any(tot==0):
        tot[tot==0] = float('nan')

    # Gini's coefficient
    coeff = totNum+1-2*(np.sum(In*freq,dim)/tot)

    # Sample correction
    if nosamplecorr:
        coeff = coeff/totNum
    else:
        coeff = coeff/(totNum-1)

    return coeff