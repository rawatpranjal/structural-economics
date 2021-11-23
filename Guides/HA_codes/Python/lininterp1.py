def lininterp1(x,y,xi):

    import numpy as np
    x = np.array(x)

    if np.size(np.where(xi<x)) == 0:
        placeLow = np.size(x)-2
    else:
        placeLow = np.where(xi<x)[0][0]-1

    if placeLow == -1:
        placeLow = 0

    placeHigh = placeLow+1
    xLow = x[placeLow]
    xHigh = x[placeHigh]
    yLow = y[placeLow]
    yHigh = y[placeHigh]

    yi = yLow+(xi-xLow)*(yHigh-yLow)/(xHigh-xLow)
    return(yi)