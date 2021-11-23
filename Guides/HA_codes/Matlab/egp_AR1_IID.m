% Endogenous Grid Points with AR1 + IID Income
% Cash on Hand as State variable
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS


% preferences
risk_aver   = 1;
beta        = 0.96;

%returns
r           = 0.02;
R = 1+ r;

% income risk: AR(1) + IID in logs
nyT         = 5; %5; %transitory component (not a state variable)
sd_logyT   = 0.2;

nyP         = 11; %9; % persistent component
sd_logyP    = 0.24; %0.1;
rho_logyP   = 0.97;

% cash on hand / savings grid
nx          = 50;
xmax        = 40; 
xgrid_par   = 0.4; %1 for linear, 0 for L-shaped
borrow_lim  = 0;

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;

%mpc options
mpcamount1  = 1.0e-10; %approximate thoeretical mpc
mpcamount2  = 0.10; % one percent of average income: approx $500
mpcamount3  = 1.0; % one percent of average income: approx $5000

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

%% SET UP GRIDS

%persistent income
% logyPgrid
% 
% [logyPgrid, yPtrans, yPdist] = rouwenhorst(nyP, -0.5*sd_logyP^2, sd_logyP, rho_logyP);
% yPgrid = exp(logyPgrid);
% yPcumdist = cumsum(yPdist);
% yPcumtrans = cumsum(yPtrans,2);
% 
% % transitory income: disretize normal distribution
% if nyT>1
%     width = fzero(@(x)discrete_normal(nyT,-0.5*sd_logyT^2 ,sd_logyT ,x),2);
%     [temp,logyTgrid,yTdist] = discrete_normal(nyT,-0.5*sd_logyT^2 ,sd_logyT ,width);
% elseif nyT==1
%     logyTgrid = 0;
%     yTdist = 1;
% end
% yTgrid = exp(logyTgrid);



%% UTILITY FUNCTION

if risk_aver==1
    u = @(c)log(c);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
end    

u1 = @(c) c.^(-risk_aver);
u1inv = @(u) u.^(-1./risk_aver);

%% INITIALIZE CONSUMPTION FUNCTION

conguess = zeros(nx,nyP);
for iy = 1:nyP
    conguess(:,iy) = r.*xgrid;
end


%% ITERATE ON EULER EQUATION WITH ENDOGENOUS GRID POINTS

con = conguess;

iter = 0;
cdiff = 1000;

while iter <= max_iter && cdiff>tol_iter
    iter = iter + 1;
    sav = zeros(nx,nyP);
    emuc= zeros(ns,nyP);
    
    conlast = con;
    muc  = u1(conlast);   %muc on grid for x' 
    
    %create interpolating function for each yP'
    for iy = 1:nyP
        if InterpMUC==1
            mucinterp{iy} = griddedInterpolant(xgrid,muc(:,iy),'linear');
        elseif InterpCon==1
            coninterp{iy} = griddedInterpolant(xgrid,conlast(:,iy),'linear');
        end    
    end

    
    % loop over current persistent income
    for iy = 1:nyP
        
        %loop over future income realizations    
        for iyP2 = 1:nyP
            for iyT2 = 1:nyT
                if InterpMUC==1
                    emuc(:,iy) = emuc(:,iy) + mucinterp{iyP2}(R.*sgrid(:) + yPgrid(iyP2)*yTgrid(iyT2)) * yPtrans(iy,iyP2) * yTdist(iyT2);
                elseif InterpCon==1
                    emuc(:,iy) = emuc(:,iy) + u1(coninterp{iyP2}(R.*sgrid(:) + yPgrid(iyP2)*yTgrid(iyT2))) * yPtrans(iy,iyP2) * yTdist(iyT2);
                end    
            end
        end
        
        muc1(:,iy) = beta.*R.*emuc(:,iy);
        con1(:,iy) = u1inv(muc1(:,iy));
        cash1(:,iy) = con1(:,iy) + sgrid;
        
        
        % loop over current period cash on hand
        for ix = 1:nx 
            if xgrid(ix)<cash1(1,iy) %borrowing constraint binds
                sav(ix,iy) = borrow_lim;
            else %borrowing constraint does not bind;
                sav(ix,iy) = lininterp1(cash1(:,iy),sgrid,xgrid(ix));
            end                
        end
        con(:,iy) = xgrid - sav(:,iy);
    end   
    
    cdiff = max(max(abs(con-conlast)));
    if Display >=1
        disp(['Iteration no. ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
    end
end    


%% SIMULATE
if DoSimulate==1

    yPindsim = zeros(Nsim,Tsim);
    logyTsim = zeros(Nsim,Tsim);
    logyPsim = zeros(Nsim,Tsim);
    ysim = zeros(Nsim,Tsim);    
    xsim = zeros(Nsim,Tsim);
    ssim = zeros(Nsim,Tsim);
    
    %create interpolating function
    for iy = 1:nyP
        savinterp{iy} = griddedInterpolant(xgrid,sav(:,iy),'linear');
    end
    
    %initialize permanent income;
    it = 1;
    yPindsim(yPrand(:,it)<= yPcumdist(1),it) = 1;
    for iy = 2:nyP
        yPindsim(yPrand(:,it)> yPcumdist(iy-1) & yPrand(:,it)<=yPcumdist(iy),it) = iy;
    end
    
    %loop over time periods
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
            disp([' Simulating, time period ' int2str(it)]);
        end
        
        %permanent income realization: note we vectorize simulations at once because
        %of matlab, in other languages we would loop over individuals
        if it > 1
            yPindsim(yPrand(:,it)<= yPcumtrans( yPindsim(:,it-1),1),it) = 1;
            for iy = 2:nyP
                yPindsim(yPrand(:,it)> yPcumtrans(yPindsim(:,it-1),iy-1) & yPrand(:,it)<=yPcumtrans(yPindsim(:,it-1),iy),it) = iy;
            end
        end
        logyPsim(:,it) = logyPgrid(yPindsim(:,it));
        
        %transitory income realization
        logyTsim(:,it) = - 0.5*sd_logyT^2 + yTrand(:,it).*sd_logyT;
        
        ysim(:,it) = exp(logyPsim(:,it) + logyTsim(:,it));
        
        %update cash on hand
        if it>1
            xsim(:,it) = R.*ssim(:,it-1) + ysim(:,it);
        end
        
        %savings choice
        if it<Tsim        
            for iy = 1:nyP
                ssim(yPindsim(:,it)==iy,it) = savinterp{iy}(xsim(yPindsim(:,it)==iy,it));
            end
        end
        ssim(ssim<borrow_lim) = borrow_lim;
		
        
    end
    
    %convert to assets
    asim = (xsim - ysim)./R;

    %consumption
    csim = xsim - ssim;

end

 
%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption policy function
    subplot(2,4,1);
    plot(xgrid,con(:,1),'b-',xgrid,con(:,nyP),'r-','LineWidth',1);
    grid;
    xlim([borrow_lim xmax]);
    title('Consumption Policy Function');
    legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(xgrid,sav(:,1)./xgrid,'b-',xgrid,sav(:,nyP)./xgrid,'r-','LineWidth',1);
    hold on;
    plot(xgrid,ones(nx,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([borrow_lim xmax]);
    title('Savings Policy Function s/x');
    
    % consumption policy function: zoomed in
    subplot(2,4,3);
    plot(xgrid,con(:,1),'b-o',xgrid,con(:,nyP),'r-o','LineWidth',2);
    grid;
    xlim([0 4]);
    title('Consumption: Zoomed');
    
     % savings policy function: zoomed in
    subplot(2,4,4);
    plot(xgrid,sav(:,1)./xgrid,'b-o',xgrid,sav(:,nyP)./xgrid,'r-o','LineWidth',2);
    hold on;
    plot(xgrid,ones(nx,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 4]);
    title('Savings (s/x): Zoomed');
    
    
    %income distribution
    subplot(2,4,5);
    hist(ysim(:,Tsim),50);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[0 0.5 0.5],'EdgeColor','blue','LineStyle','-');
    ylabel('')
    title('Income distribution');
    
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
    
    
   % asset distribution statistics
    aysim = asim(:,Tsim) ./ mean(ysim(:,Tsim));
    disp(['Mean assets  (relative to mean income) : ' num2str(mean(aysim))]);
    disp(['Fraction borrowing constrained: ' num2str(sum(aysim==borrow_lim)./Nsim * 100) '%']);
    disp(['10th Percentile: ' num2str(quantile(aysim,.1))]);
    disp(['50th Percentile: ' num2str(quantile(aysim,.5))]);
    disp(['90th Percentile: ' num2str(quantile(aysim,.9))]);
    disp(['99th Percentile: ' num2str(quantile(aysim,.99))]);

end

%% COMPUTE MPCs
if ComputeMPC ==1
    
    %theoretical mpc lower bound
    mpclim = R*((beta*R)^-(1./risk_aver))-1;
    
    for iy = 1:nyP
        %create interpolating function
        coninterp{iy} = griddedInterpolant(xgrid,con(:,iy),'linear');
        
        mpc1(:,iy) = ( coninterp{iy}(xgrid+mpcamount1) - con(:,iy) ) ./ mpcamount1;
        mpc2(:,iy) = ( coninterp{iy}(xgrid+mpcamount2) - con(:,iy) ) ./ mpcamount2;
        mpc3(:,iy) = ( coninterp{iy}(xgrid+mpcamount3) - con(:,iy) ) ./ mpcamount3;
        
    end

    
    figure(2);
    
    % mpc functions
    subplot(1,2,1);
    plot(xgrid,mpc1(:,(nyP+1)/2),'b-',xgrid,mpc2(:,(nyP+1)/2),'r:',xgrid,mpc3(:,(nyP+1)/2),'k--','LineWidth',1);
    hold on;
    plot(xgrid,mpclim.*ones(size(xgrid)),'k:','LineWidth',2);    
    hold off;
    grid on;
    xlim([0 10]);
    title('MPC Function: median persistant income state');
    legend('Amount 1','Amount 2','Amount 3',...        
        ['Theoretical MPC = ', num2str(mpclim)],'Location','NorthEast');
    
    % mpc distribution
    mpc1sim = zeros(Nsim,1);
    mpc2sim = zeros(Nsim,1);
    mpc3sim = zeros(Nsim,1);
    for iy = 1:nyP
        mpc1sim(yPindsim(:,Tsim)==iy) = ( coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)+mpcamount1) - coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)) ) ./ mpcamount1;
        mpc2sim(yPindsim(:,Tsim)==iy) = ( coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)+mpcamount2) - coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)) ) ./ mpcamount2;
        mpc3sim(yPindsim(:,Tsim)==iy) = ( coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)+mpcamount3) - coninterp{iy}(xsim(yPindsim(:,Tsim)==iy,Tsim)) ) ./ mpcamount3;
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
    disp(['Mean MPC amount 3: ' num2str(mean(mpc3sim))]);

    
end    
    
    
    
    
    
    
    
    
    
    
