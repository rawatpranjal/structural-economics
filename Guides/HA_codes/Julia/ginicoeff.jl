using Logging
function ginicoeff(Dat::AbstractMatrix;dim=1,nosamplecorr=false)
    
    # Negative values or one-element series (not admitted)
    IDXnan = isnan.(Dat)
    IDX    = any(Dat.<0,dims=dim) .| (sum(.!IDXnan,dims=dim).<2)
    if dim == 1
        Dat[:,IDX[:]] .= 0
    else
        Dat[IDX[:],:] .= 0
    end

    if any(IDX)
        @warn "Check IDX for negative values or one-element series " * string(IDX)
    end

    # Total numel
    totNum = sum(.!IDXnan,dims=dim)

    # Replace NaNs
    Dat[IDXnan] .= 0

    # Sort In
    Dat = sort(Dat,dims=dim)

    # Calculate frequencies for each series
    freq = reverse(cumsum(ones(size(Dat)),dims=dim),dims=dim)

    # Totals
    tot = sum(Dat,dims=dim)

    # Gini's coefficient
    coeff = totNum .+ 1 .- 2 .*(sum(Dat.*freq,dims=dim)./tot)

    # Sample correction
    if nosamplecorr
        coeff = coeff./totNum
    else
        coeff = coeff./(totNum .-1 )
    end

    return coeff
end

ginicoeff(Dat::AbstractVector;nosamplecorr=false) = ginicoeff(reshape(Dat,length(Dat),1),nosamplecorr=false)[1]