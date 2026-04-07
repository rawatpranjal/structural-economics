function shock = gen_discrete_markov_rn(trans, numPaths, lenPath, initShock, seed)

if nargin>=5
    rng(seed);
end

numStates = size(trans,1);

shock = zeros(numPaths,lenPath);
shock(:,1) = initShock;

cumTrans = cumsum(trans, 2);
un = rand(numPaths,lenPath);

for t=1:lenPath-1
    for j=1:numStates
        % Find samples that has shock j
        idxOfShockJ = find(shock(:,t)==j);
        % Generate random number ~ Un[0,1]
        % un = rand(length(idxOfShockJ),1);
        % Look up trans_j
        [~,shock(idxOfShockJ,t+1)] = histc(un(idxOfShockJ,t), [0 cumTrans(j,:)]);
    end
end
end
