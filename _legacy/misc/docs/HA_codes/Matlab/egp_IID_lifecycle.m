% Lifecycle Model with Retirement
% Endogenous Grid Points with IID Income
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS

% demographics
Twork = 40; %25-64
Tret  = 35; %65 - 99
T = Twork + Tret;

surprob = ones(T,1);
temp = load('surprobsmooth.txt');
surprob(Twork+1:T) = temp(1:Tret);

popsize = ones(T,1);
popsize(Twork+2:T) = cumprod(surprob(Twork+1:T-1));

% preferences
risk_aver   = 2;
beta        = 0.97;

%returns
r           = 0.04;
R = 1+ r;

% mean income profile
kappa = 0.1.*[1:Twork] - 0.002.*[1:Twork].^2;

% income risk: discretized N(mu,sigma^2)
mu_y    = 1;
sd_y    = 0.1;
ny      = 5;

%pension replacement rate: multiple of income in last working period
penreplace = 0.6;

% asset grids
na          = 40;
amax        = 50; 
borrow_lim  = 0;
agrid_par   = 0.5; %1 for linear, 0 for L-shaped

% computation
Nsim        = 50000;

%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

% which function to interpolation 
InterpCon = 0;
InterpEMUC = 1;

%tolerance for non-linear solver
options = optimset('Display','Off','TolX',1.0e-6);

%% DRAW RANDOM NUMBERS
rng(2017);
yrand = rand(Nsim,Twork);

%% SET UP GRIDS

% assets
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

% income: disretize normal distribution
width = fzero(@(x)discrete_normal(ny,mu_y,sd_y,x),2);
[temp,ygrid,ydist] = discrete_normal(ny,mu_y,sd_y,width);
ycumdist = cumsum(ydist);

% pension
pengrid = penreplace .* exp(kappa(Twork)).* ygrid;

%grids on (a,y) space
aagrid = repmat(agrid,1,ny);


%% UTILITY FUNCTION

u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
u1 = @(c) c.^(-risk_aver);
u1inv = @(u) u.^(-1./risk_aver);

%% INITIALIZE ARRAYS
con = zeros(na,ny,Twork);
sav = zeros(na,ny,Twork);

conret = zeros(na,ny,Tret);
savret = zeros(na,ny,Tret);

% emuc = zeros(na,ny);
% muc1 = zeros(na,ny);
% con1 = zeros(na,ny);
% ass1 = zeros(na,ny);
% emuc = zeros(na,ny);

%% DECISIONS AT t=T
for iy = 1:ny
    conret(:,iy,Tret) = R.*agrid +pengrid(iy) - sav(:,iy);
end
savret(:,:,Tret) = 0;

%% SOlVE EULER EQUATION BACKWARD WITH ENDOGENOUS GRID POINTS

%retirement period
for it = Tret-1:-1:1
    if Display >=1 
        disp(['Solving at age: ' int2str(Twork +it)]);
    end
    
    emucret = u1(conret(:,:,it+1)); 
    muc1ret = beta.*R.*surprob(Twork+it).*emucret; 
    con1ret = u1inv(muc1ret);
    
    % loop over pension grid
    for iy = 1:ny
        
        ass1ret(:,iy) = (con1ret(:,iy) + agrid -pengrid(iy))./R;
        
        % loop over current period ssets
        for ia  = 1:na 
            if agrid(ia)<ass1ret(1,iy) %borrowing constraint binds
                savret(ia,iy,it) = borrow_lim;
            else %borrowing constraint does not bind;
                savret(ia,iy,it) = lininterp1(ass1ret(:,iy),agrid,agrid(ia));
            end                
        end
        conret(:,iy,it) = R.*agrid +pengrid(iy) - savret(:,iy,it);
    end   
    
end    


%final working period
it = Twork;
if Display >=1 
    disp(['Solving at age: ' int2str(it)]);
end

emucret = u1(conret(:,:,1)); 
muc1ret = beta.*R.*surprob(it).*emucret; 
con1ret = u1inv(muc1ret);

% loop over pension grid
for iy = 1:ny

    ass1(:,iy) = (con1ret(:,iy) + agrid - exp(kappa(it)).*ygrid(iy))./R;

    % loop over current period ssets
    for ia  = 1:na 
        if agrid(ia)<ass1(1,iy) %borrowing constraint binds
            sav(ia,iy,it) = borrow_lim;
        else %borrowing constraint does not bind;
            sav(ia,iy,it) = lininterp1(ass1(:,iy),agrid,agrid(ia));
        end                
    end
    con(:,iy,it) = R.*agrid + exp(kappa(it)).*ygrid(iy) - sav(:,iy,it);
end   


%other working periods
for it = Twork-1:-1:1
    
    if Display >=1 
        disp(['Solving at age: ' int2str(it)]);
    end

        
    emuc = u1(squeeze(con(:,:,it+1))) * ydist; 
    muc1 = beta.*R.*surprob(it).*emuc; 
    con1 = u1inv(muc1);
    
    % loop over pension grid
    for iy = 1:ny
        
        ass1(:,iy) = (con1 + agrid - exp(kappa(it)).*ygrid(iy))./R;
        
        % loop over current period ssets
        for ia  = 1:na 
            if agrid(ia)<ass1(1,iy) %borrowing constraint binds
                sav(ia,iy,it) = borrow_lim;
            else %borrowing constraint does not bind;
                sav(ia,iy,it) = lininterp1(ass1(:,iy),agrid,agrid(ia));
            end                
        end
        con(:,iy,it) = R.*agrid + exp(kappa(it)).*ygrid(iy) - sav(:,iy,it);
    end   
    

end    


%% SIMULATE
if DoSimulate==1

    yindsim = zeros(Nsim,T);
    asim = zeros(Nsim,T+1);
    
    %loop over time periods
    for it = 1:T

        if Display >=1 
            disp(['Simulating at age: ' int2str(it)]);
        end

        %create interpolating function
        for iy = 1:ny
            if it<=Twork
                savinterp{it,iy} = griddedInterpolant(agrid,sav(:,iy,it),'linear');
            elseif it >Twork
                savinterp{it,iy} = griddedInterpolant(agrid,savret(:,iy,it-Twork),'linear');
            end
        end
            
        %income realization: note we vectorize simulations at once because
        %of matlab, in other languages we would loop over individuals
        if it<=Twork
            yindsim(yrand(:,it)<= ycumdist(1),it) = 1;
            for iy = 2:ny
                yindsim(yrand(:,it)> ycumdist(iy-1) & yrand(:,it)<=ycumdist(iy),it) = iy;
            end
        elseif it >Twork
            yindsim(:,it) = yindsim(:,it-1);
        end
        
        % asset choice
        for iy = 1:ny
            asim(yindsim(:,it)==iy,it+1) = savinterp{it,iy}(asim(yindsim(:,it)==iy,it));
        end
        
        %assign actual income values;
        if it<=Twork
            ysim(:,it) = exp(kappa(it)).* ygrid(yindsim(:,it));
        elseif it>Twork
            ysim(:,it) = pengrid(yindsim(:,it));
        end
        
        %consumption
        csim(:,it) = R.*asim(:,it) + ysim(:,it) - asim(:,it+1);

    end    

end


 
%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % mean income and consumption
    subplot(2,2,1);
    plot([1:T],mean(ysim,1),'k-',[1:T],mean(csim,1),'b--','LineWidth',2);   
    grid;
    ylim([0 4]);
    legend('Income, pension', 'Consumtion','Location','NorthEast');
    title('Mean: income, consumption');
    
    % mean assets
    subplot(2,2,2);
    plot([1:T+1],mean(asim,1),'r-','LineWidth',2);   
    grid;
    title('Mean: assets');

     % variance log income and log consumption
    subplot(2,2,3);
    plot([1:T],var(log(ysim),1),'k-',[1:T],var(log(csim),1),'b--','LineWidth',2);   
    grid;
%     ylim([0 4]);
%     legend('Income, pension', 'Consumtion','Location','NorthEast');
    title('Var log: income, consumption');
    
    % population statistics
    Ea_byage = mean(asim,1);
    Ey_byage = mean(ysim,1);
    Ec_byage = mean(csim,1);
    Ea0_byage = mean(asim==0,1);
    
    Ea = sum(Ea_byage(1:T).*popsize') ./sum(popsize);
    Ey = sum(Ey_byage.*popsize') ./sum(popsize);
    Ec = sum(Ec_byage.*popsize') ./sum(popsize);
    Ea0 = sum(Ea0_byage(1:T).*popsize') ./sum(popsize);
   
    disp(['Mean assets: ' num2str(Ea)]);
    disp(['Mean income: ' num2str(Ey)]);
    disp(['Mean consumption: ' num2str(Ec)]);
    disp(['Fraction borrowing constrained: ' num2str(Ea0*100) '%']);

end
