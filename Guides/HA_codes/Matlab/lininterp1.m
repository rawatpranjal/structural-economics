function [yi] = lininterp1(x,y,xi)
%[yi] = lininterp1(x,y,xi)
%x is N x 1
%y is N x 1
%xi is 1 x 1
%yi is 1 x 1
%same as lininterp except is faster and only looks up one function
%extrapolates out of range

placeLow = find(xi<x,1)-1;
if placeLow == 0
    placeLow = 1;
end
if isempty(placeLow)
    placeLow = length(x)-1;
end    
placeHigh = placeLow+1;
xLow    = x(placeLow);
xHigh   = x(placeHigh);
yLow    = y(placeLow);
yHigh    = y(placeHigh);

yi = yLow +(xi-xLow).*(yHigh-yLow)./(xHigh-xLow);