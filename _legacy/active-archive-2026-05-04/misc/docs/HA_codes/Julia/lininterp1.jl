function lininterp1(x,y,xi)

    x = x[:]
    
    if isnothing(findfirst(xi.<x))
        placeLow = length(x)-1
    else
        placeLow = findfirst(xi.<x)-1
    end
    
    if placeLow == 0
        placeLow = 1
    end
    
    placeHigh = placeLow+1
    xLow = x[placeLow]
    xHigh = x[placeHigh]
    yLow = y[placeLow]
    yHigh = y[placeHigh]

    yi = yLow +(xi-xLow)*(yHigh-yLow)/(xHigh-xLow)
    
end