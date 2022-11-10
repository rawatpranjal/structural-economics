function [grid, trans, dist] = rouwenhorst(n, mu, sigma, rho)
    
    % grid
    width = sqrt((n-1) * sigma^2 / ( 1 - rho^2));
    grid = linspace( mu-width, mu + width, n)';
    
    %transition matrix
    p0 = (1 + rho) / 2;
    trans = [p0 1-p0; 1-p0 p0];
    
    if n > 2
        for i = 1:n-2
            cstr_temp = zeros(length(trans(:,1)), 1);
            trans = p0 .* [trans cstr_temp; cstr_temp.' 0] + (1 - p0 ) .* [cstr_temp trans; cstr_temp.' 0]  + (1 - p0 ) .*  [ cstr_temp.' 0; trans cstr_temp] + p0 .* [ cstr_temp.' 0; cstr_temp trans];
        end
        trans = bsxfun(@rdivide, trans, sum(trans,2));
    end
    
    % ergodic distribution
    dist = ones(1,n)./n;
    for i = 1: 100
        dist = dist*(trans^i);
    end
    dist = dist';
end
    