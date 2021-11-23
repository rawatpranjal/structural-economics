% Endogenous Grid Points with AR1 + IID Income
% Cash on Hand as State variable
% Includes NIT and discount factor heterogeneity
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS


% preferences
risk_aver   = 1;
beta        = 0.955; %0.955;

%warm glow bequests: bequest_weight = 0 is accidental
bequest_weight = 0; %0.07;
bequest_luxury = 2; %0.01;
dieprob     = 0; %1/50;

%returns
r           = 0.02;
R = 1+ r;

% income risk: AR(1) + IID in logs
nyT         = 5; %transitory component (not a state variable)
sd_logyT   = 0.20; %relevant if nyT>1

nyP         = 11; %9; % persistent component
sd_logyP    = 0.24; %
rho_logyP   = 0.97;

% cash on hand / savings grid
nx          = 50;
xmax        = 40; 
xgrid_par   = 0.4; %1 for linear, 0 for L-shaped
borrow_lim  = 0;

%government
labtaxlow       = 0; %proportional tax
labtaxhigh      = 0; %additional tax on incomes above threshold
labtaxthreshpc   = 0.99; %percentile of earnings distribution where high tax rate kicks in

savtax          = 0.0001;  %tax rate on savings
savtaxthresh    = 0; %multiple of mean gross labor income


%discount factor shocks;
nb = 1;  %1 or 2
betawidth = 0.065; % beta +/- beta width
betaswitch = 1/40; %0;

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 100000;
Tsim        = 300;

%mpc options
mpcamount1  = 1.0e-10; %approximate thoeretical mpc
mpcamount2  = 0.10; % one percent of average income: approx $500

%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;
ComputeMPC  = 1;

% which function to interpolation 
InterpMUC = 0;
InterpCon = 1; %sometimes useful to stop extrapolating to negative MUC


%% DRAW RANDOM NUMBERS
rng(2017);
yPrand = rand(Nsim,Tsim);
yTrand = randn(Nsim,Tsim);
betarand = rand(Nsim,Tsim);
dierand = rand(Nsim,Tsim);

%% INCOME GRIDS

%persistent income: rowenhurst
[logyPgrid, yPtrans, yPdist] = rouwenhorst(nyP, -0.5*sd_logyP^2, sd_logyP, rho_logyP);
yPgrid = exp(logyPgrid);
yPcumdist = cumsum(yPdist);
yPcumtrans = cumsum(yPtrans,2);

% transitory income: disretize normal distribution
if nyT>1
    width = fzero(@(x)discrete_normal(nyT,-0.5*sd_logyT^2 ,sd_logyT ,x),2);
    [temp,logyTgrid,yTdist] = discrete_normal(nyT,-0.5*sd_logyT^2 ,sd_logyT ,width);
elseif nyT==1
    logyTgrid = 0;
    yTdist = 1;
end
yTgrid = exp(logyTgrid);

%gross labor income grid;
grosslabincgrid = zeros(nyP,nyT);
grosslabincdist= zeros(nyP,nyT);
for iyP = 1:nyP
    for iyT= 1:nyT
        grosslabincgrid(iyP,iyT) = yPgrid(iyP)*yTgrid(iyT);
        grosslabincdist(iyP,iyT) = yPdist(iyP)*yTdist(iyT);
    end
end    

%gross labor income distribution;
grosslabincgridvec = grosslabincgrid(:);
grosslabincdistvec = grosslabincdist(:);
temp = sortrows([grosslabincgrid(:) grosslabincdist(:)],1);
glincgridvec = temp(:,1);
glincdistvec = temp(:,2);
glincdistcumvec = cumsum(glincdistvec);


% labor taxes and transfers function
meangrosslabinc = sum(glincgridvec.*glincdistvec);
totgrosslabinc = meangrosslabinc;
labtaxthresh = lininterp1(glincdistcumvec,glincgridvec,labtaxthreshpc);
totgrosslabinchigh = sum(max(glincgridvec - labtaxthresh,0).*glincdistvec);

lumptransfer  = labtaxlow*totgrosslabinc + labtaxhigh*totgrosslabinchigh ;

%net labor income
netlabincgrid  = lumptransfer + (1-labtaxlow).*grosslabincgrid - labtaxhigh.*max(grosslabincgrid- labtaxthresh,0);



%% ASSET GRIDS

% cash on hand grids: different min points for each value of (iyP)
for iyP = 1:nyP
    xgrid(:,iyP) = linspace(0,1,nx)';
    xgrid(:,iyP) = xgrid(:,iyP).^(1./xgrid_par);
    xgrid(:,iyP) = borrow_lim + min(netlabincgrid(iyP,:)) + (xmax-borrow_lim).*xgrid(:,iyP);
end

% savings grid for EGP
ns = nx;
sgrid = linspace(0,1,nx)';
sgrid = sgrid.^(1./xgrid_par);
sgrid = borrow_lim + (xmax-borrow_lim).*sgrid;

%discount factor grid
if  nb == 1
    betagrid = beta;
    betadist = 1;
    betatrans = 1;
elseif nb ==2 
    betagrid = [beta-betawidth;beta+betawidth];
    betadist = [0.5;0.5];
    betatrans = [1-betaswitch betaswitch; betaswitch 1-betaswitch]; %transitions on average once every 40 years;
else
    error('nb must be 1 or 2');
end
betacumdist = cumsum(betadist);
betacumtrans = cumsum(betatrans,2);



%% UTILITY FUNCTION, BEQUEST FUNCTION

if risk_aver==1
    u = @(c)log(c);
    beq = @(a) bequest_weight.* log(a+ bequest_luxury);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
    beq = @(a) bequest_weight.*((a+bequest_luxury).^(1-risk_aver)-1)./(1-risk_aver);
end    

u1 = @(c) c.^(-risk_aver);
u1inv = @(u) u.^(-1./risk_aver);

beq1 = @(a) bequest_weight.*(a+bequest_luxury).^(-risk_aver);


%% INITIALIZE CONSUMPTION FUNCTION

conguess = zeros(nx,nyP,nb);
for ib = 1:nb
for iy = 1:nyP
    conguess(:,iy,ib) = r.*xgrid(:,iy);
end
end

%% ITERATE ON EULER EQUATION WITH ENDOGENOUS GRID POINTS

con = conguess;

iter = 0;
cdiff = 1000;

while iter <= max_iter && cdiff>tol_iter
    iter = iter + 1;
    sav = zeros(nx,nyP,nb);
    emuc= zeros(ns,nyP,nb);
    
    conlast = con;
    muc  = u1(conlast);   %muc on grid for x' 
    
    %create interpolating function for each yP'
    for ib = 1:nb
    for iy = 1:nyP
        if InterpMUC==1
            mucinterp{iy,ib} = griddedInterpolant(xgrid(:,iy),muc(:,iy,ib),'linear');
        elseif InterpCon==1
            coninterp{iy,ib} = griddedInterpolant(xgrid(:,iy),conlast(:,iy,ib),'linear');
        end    
    end
    end
    
   
    % loop over current persistent income and discount factor
    for ib = 1:nb
    for iy = 1:nyP
        
        %loop over future income realizations    and discount factor
        for ib2 = 1:nb
            for iyP2 = 1:nyP
                for iyT2 = 1:nyT
                    if InterpMUC==1
                        emuc(:,iy,ib) = emuc(:,iy,ib) + mucinterp{iyP2,ib2}(R.*sgrid(:) + netlabincgrid(iyP2,iyT2)) * yPtrans(iy,iyP2) * yTdist(iyT2) * betatrans(ib,ib2);
                    elseif InterpCon==1
                        emuc(:,iy,ib) = emuc(:,iy,ib) + u1(coninterp{iyP2,ib2}(R.*sgrid(:) + netlabincgrid(iyP2,iyT2))) * yPtrans(iy,iyP2) * yTdist(iyT2) * betatrans(ib,ib2);
                    end    
                end
            end
        end
        muc1(:,iy,ib) = (1-dieprob) .*  betagrid(ib).*R.*emuc(:,iy,ib)./(1+savtax.*(sgrid>=savtaxthresh)) + dieprob.*beq1(sgrid);
        con1(:,iy,ib) = u1inv(muc1(:,iy,ib));
        cash1(:,iy,ib) = con1(:,iy,ib) + sgrid + savtax.*sgrid.*(sgrid>=savtaxthresh);
        
        
        % loop over current period cash on hand
        for ix = 1:nx 
            if xgrid(ix,iy)<cash1(1,iy,ib) %borrowing constraint binds
                sav(ix,iy,ib) = borrow_lim;
            else %borrowing constraint does not bind;
                sav(ix,iy,ib) = lininterp1(cash1(:,iy,ib),sgrid,xgrid(ix,iy));
            end                
        end
        con(:,iy,ib) = xgrid(:,iy) - sav(:,iy,ib);
    end   
    end
    
    cdiff = max(abs(con(:)-conlast(:)));
    if Display >=1
        disp(['Iteration no. ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
    end
end    


%% SIMULATE
if DoSimulate==1

    
    yPindsim = zeros(Nsim,Tsim);
    logyTsim = zeros(Nsim,Tsim);
    logyPsim = zeros(Nsim,Tsim);
    ygrosssim = zeros(Nsim,Tsim);    
    ynetsim = zeros(Nsim,Tsim);    
    xsim = zeros(Nsim,Tsim);
    ssim = zeros(Nsim,Tsim);
    betaindsim = zeros(Nsim,Tsim);
    betasim = zeros(Nsim,Tsim);
    
    %indicators for dying
    diesim = dierand<dieprob;
        
    
    %create interpolating functions
    for ib = 1:nb
    for iy = 1:nyP
        savinterp{iy,ib} = griddedInterpolant(xgrid(:,iy),sav(:,iy,ib),'linear');
    end
    end
    
    %initialize permanent income;
    it = 1;
    yPindsim(yPrand(:,it)<= yPcumdist(1),it) = 1;
    for iy = 2:nyP
        yPindsim(yPrand(:,it)> yPcumdist(iy-1) & yPrand(:,it)<=yPcumdist(iy),it) = iy;
    end
    
    %initialize discount factor;
    it = 1;
    betaindsim(betarand(:,it)<= betacumdist(1),it) = 1;
    for ib = 2:nb
        betaindsim(betarand(:,it)> betacumdist(ib-1) & betarand(:,it)<=betacumdist(ib),it) = ib;
    end
    
    
    %loop over time periods
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
            disp([' Simulating, time period ' int2str(it)]);
        end
        
        
        %permanent income realization: note we vectorize simulations at once because
        %of matlab, in other languages we would loop over individuals
        
        %people who dont die: use transition matrix
        if it > 1
            yPindsim(diesim(:,it)==0 & yPrand(:,it)<= yPcumtrans( yPindsim(:,it-1),1),it) = 1;
            for iy = 2:nyP
                yPindsim(diesim(:,it)==0 & yPrand(:,it)> yPcumtrans(yPindsim(:,it-1),iy-1) & yPrand(:,it)<=yPcumtrans(yPindsim(:,it-1),iy),it) = iy;
            end
        end
        
        %people who die: descendendents draw income from stationary distribution
        yPindsim(diesim(:,it)==1 & yPrand(:,it)<= yPcumdist(1),it) = 1;
        for iy = 2:nyP
            yPindsim(diesim(:,it)==1 & yPrand(:,it)> yPcumdist(iy-1) & yPrand(:,it)<=yPcumdist(iy),it) = iy;
        end

        logyPsim(:,it) = logyPgrid(yPindsim(:,it));
        
        %transitory income realization
        if nyT>1
            logyTsim(:,it) = - 0.5*sd_logyT^2 + yTrand(:,it).*sd_logyT;
        else
            logyTsim(:,it) = 0;
        end
        ygrosssim(:,it) = exp(logyPsim(:,it) + logyTsim(:,it));
        ynetsim(:,it) = lumptransfer + (1-labtaxlow).*ygrosssim(:,it) - labtaxhigh.*max(ygrosssim(:,it)-labtaxthresh,0);
        
        %discount factor realization: note we vectorize simulations at once because
        %of matlab, in other languages we would loop over individuals
        if it > 1
            betaindsim(betarand(:,it)<= betacumtrans( betaindsim(:,it-1),1),it) = 1;
            for ib = 2:nb
                betaindsim(betarand(:,it)> betacumtrans(betaindsim(:,it-1),ib-1) & betarand(:,it)<=betacumtrans(betaindsim(:,it-1),ib),it) = ib;
            end
        end
        
        
        
        %update cash on hand
        if it>1
            xsim(:,it) = R.*ssim(:,it-1) + ynetsim(:,it);
        end
        
        %savings choice
        for ib = 1:nb
        for iy = 1:nyP
            ssim(yPindsim(:,it)==iy & betaindsim(:,it)==ib,it) = savinterp{iy,ib}(xsim(yPindsim(:,it)==iy & betaindsim(:,it)==ib,it));
        end
        end
        ssim(ssim<borrow_lim) = borrow_lim;
    end
    
    %convert to assets
    asim = (xsim - ynetsim)./R;

    %consumption
    csim = xsim - ssim - savtax.*max(ssim-savtaxthresh,0);
end

 
%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption policy function
    subplot(2,4,1);
    plot(xgrid(:,1),con(:,1),'b-',xgrid(:,nyP),con(:,nyP),'r-','LineWidth',1);
    grid;
    xlim([borrow_lim xmax]);
    title('Consumption Policy Function');
    legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(xgrid(:,1),sav(:,1)./xgrid(:,1),'b-',xgrid(:,nyP),sav(:,nyP)./xgrid(:,nyP),'r-','LineWidth',1);
    hold on;
    plot(sgrid,ones(nx,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([borrow_lim xmax]);
    title('Savings Policy Function s/x');
    
    % consumption policy function: zoomed in
    subplot(2,4,3);
    plot(xgrid(:,1),con(:,1),'b-o',xgrid(:,nyP),con(:,nyP),'r-o','LineWidth',2);
    grid;
    xlim([0 4]);
    title('Consumption: Zoomed');
    
     % savings policy function: zoomed in
    subplot(2,4,4);
    plot(xgrid(:,1),sav(:,1)./xgrid(:,1),'b-o',xgrid(:,nyP),sav(:,nyP)./xgrid(:,nyP),'r-o','LineWidth',2);
    hold on;
    plot(sgrid,ones(nx,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 4]);
    title('Savings (s/x): Zoomed');
    
    
    %income distribution
    subplot(2,4,5);
    hist(ygrosssim(:,Tsim),50);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[0 0.5 0.5],'EdgeColor','blue','LineStyle','-');
    ylabel('')
    title('Gross Income distribution');
    
    %asset distribution
    subplot(2,4,6:7);
    hist(asim(:,Tsim),100);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[.7 .7 .7],'EdgeColor','black','LineStyle','-');
    ylabel('')
    title('Asset distribution');

    %convergence check
    subplot(2,4,8);
    plot([1:Tsim]',mean(asim,1),'k-','LineWidth',1.5);
    ylabel('Time Period');
    title('Mean Asset Convergence');
    
    
   % distribution statistics
    aysim = asim(:,Tsim) ./ meangrosslabinc;
    disp(['Mean assets  (relative to mean income) : ' num2str(mean(aysim))]);
    disp(['Fraction borrowing constrained: ' num2str(sum(aysim==borrow_lim)./Nsim * 100) '%']);
    disp(['10th Percentile: ' num2str(quantile(aysim,.1))]);
    disp(['50th Percentile: ' num2str(quantile(aysim,.5))]);
    disp(['90th Percentile: ' num2str(quantile(aysim,.9))]);
    disp(['99th Percentile: ' num2str(quantile(aysim,.99))]);

    %table of parameters;
    tabparam  = zeros(4,1);
    tabparam(1) = beta; %discount factor
    tabparam(2) = risk_aver; %risk aversion
    tabparam(3) = labtaxlow; %labor tax rate
    tabparam(4) = lumptransfer./meangrosslabinc; %relative to mean gross lab inc
    
    %table of statistics
    tabstat = zeros(14,1);
    tabstat(1) = var(log(ygrosssim(:,Tsim))); %variance log gross earnings;
    tabstat(2) = ginicoeff(ygrosssim(:,Tsim)); %gini gross earnings;
    tabstat(3) = var(log(ynetsim(:,Tsim))); %variance log net earnings;
    tabstat(4) = ginicoeff(ynetsim(:,Tsim)); %gini net earnings;
    tabstat(5) = var(log(csim(:,Tsim))); %variance log consumption;
    tabstat(6) = ginicoeff(csim(:,Tsim)); %gini consumption;
    tabstat(7) = mean(aysim); %mean wealth (relative to mean earnings);
    tabstat(8) = quantile(aysim,.5); %median wealth (relative to mean earnings);
    tabstat(9) = ginicoeff(aysim); %wealth gini;
    tabstat(10) = quantile(aysim,.9) ./ quantile(aysim,.5); %90-10 wealth distribution
    tabstat(11) = quantile(aysim,.99) ./ quantile(aysim,.5); %99-50 wealth distribution
    tabstat(12) = sum(aysim<=0)./Nsim; %fraction  less than equal to zero;
    tabstat(13) = sum(aysim<=0.05)./Nsim; %fraction  less than equal to 5% av. annual gross earnings;
    tabstat(14) = sum(aysim(aysim>=quantile(aysim,.9))) ./ sum(aysim); %top 10% wealth share;
    tabstat(15) = sum(aysim(aysim>=quantile(aysim,.99))) ./ sum(aysim); %top 1% wealth share;
    tabstat(16) = sum(aysim(aysim>=quantile(aysim,.999))) ./ sum(aysim); %top 0.1% wealth share;
    
    taboutput = [tabparam; tabstat];
end

%% COMPUTE MPCs
if ComputeMPC ==1
    
    %theoretical mpc lower bound
    mpclim = R*((beta*R)^-(1./risk_aver))-1;
    
    for iy = 1:nyP
        %create interpolating function
        coninterp{iy} = griddedInterpolant(xgrid(:,iy),con(:,iy),'linear');
        
        mpc1(:,iy) = ( coninterp{iy}(xgrid(:,iy)+mpcamount1) - con(:,iy) ) ./ mpcamount1;
        mpc2(:,iy) = ( coninterp{iy}(xgrid(:,iy)+mpcamount2) - con(:,iy) ) ./ mpcamount2;
        
    end

    
    figure(2);
    
    % mpc functions
    subplot(1,2,1);
    plot(xgrid,mpc1(:,(nyP+1)/2),'b-',xgrid,mpc2(:,(nyP+1)/2),'b--','LineWidth',1);
    hold on;
    plot(xgrid,mpclim.*ones(size(xgrid)),'k:','LineWidth',2);    
    hold off;
    grid on;
    xlim([0 10]);
    title('MPC Function: median persistant income state');
    legend('Amount 1','Amount 2',...        
        ['Theoretical MPC limit = ', num2str(mpclim)],'Location','NorthEast');
    
    % mpc distribution
    mpc1sim = zeros(Nsim,1);
    mpc2sim = zeros(Nsim,1);
    for iy = 1:nyP
        mpc1sim(yPindsim(:,Tsim)==iy) = ( coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)+mpcamount1) - coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)) ) ./ mpcamount1;
        mpc2sim(yPindsim(:,Tsim)==iy) = ( coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)+mpcamount2) - coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)) ) ./ mpcamount2;
    end
    
    subplot(1,2,2);
    histogram(mpc1sim,[0:0.02:1.5]);
    grid on;
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[.7 .7 .7],'EdgeColor','black','LineStyle','-');
    ylabel('')
    title('MPC distribution');
    
    
    % mpc distribution statistics
    disp(['Mean MPC amount 1: ' num2str(mean(mpc1sim))]);
    disp(['Mean MPC amount 2: ' num2str(mean(mpc2sim))]);

    
end    
    
    
    
    
    
    
    
    
    
    
