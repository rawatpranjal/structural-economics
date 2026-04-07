% Aiyagari model
% Endogenous Grid Points with IID Income
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS

% preferences
risk_aver   = 2;
beta        = 0.95;

%production
deprec      = 0.10;
capshare    = 0.4;

% income risk: discretized N(mu,sigma^2)
mu_y    = 1;
sd_y    = 0.2;
ny      = 5;

% asset grids
na          = 40;
amax        = 50; 
borrow_lim  = 0;
agrid_par   = 0.5; %1 for linear, 0 for L-shaped

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;

maxiter_KL  = 70;
tol_KL      = 1.0e-5;
step_KL     = 0.005;
rguess      = 1./beta-1 - 0.001; % a bit lower than inverse of discount rate
KLratioguess = ((rguess + deprec)/capshare)^(1/(capshare - 1));


%% OPTIONS
Display     = 1;
MakePlots   = 1;

% which function to interpolation 
InterpCon = 0;
InterpEMUC = 1;

%tolerance for non-linear solver
options = optimset('Display','Off','TolX',1.0e-6);

%% UTILITY FUNCTION

if risk_aver==1
    u = @(c)log(c);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
end    

u1 = @(c) c.^(-risk_aver);
u1inv = @(u) u.^(-1./risk_aver);


%% SET UP GRIDS

% assets
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

% income: disretize normal distribution
width = fzero(@(x)discrete_normal(ny,mu_y,sd_y,x),2);
[temp,ygrid,ydist] = discrete_normal(ny,mu_y,sd_y,width);
ycumdist = cumsum(ydist);

%% DRAW RANDOM NUMBERS
rng(2017);
yrand = rand(Nsim,Tsim);

%% SIMULATE LABOR EFFICIENCY REALIZATIONS
if Display >=1
    disp(['Simulating labor efficiency realizations in advance']);
end
yindsim = zeros(Nsim,Tsim);
    
for it = 1:Tsim

    %income realization: note we vectorize simulations at once because
    %of matlab, in other languages we would loop over individuals
    yindsim(yrand(:,it)<= ycumdist(1),it) = 1;
    for iy = 2:ny
        yindsim(yrand(:,it)> ycumdist(iy-1) & yrand(:,it)<=ycumdist(iy),it) = iy;
    end
end
    
ysim = ygrid(yindsim);

%% ITERATE OVER KL RATIO
KLratio = KLratioguess;

iterKL = 0;
KLdiff = 1;

while iterKL <= maxiter_KL && abs(KLdiff)>tol_KL
    iterKL = iterKL + 1;

    r   = capshare.*KLratio^(capshare-1) - deprec;
    R   = 1+r;
    wage= (1-capshare).* KLratio^capshare;

    % rescale efficiency units of labor so that output = 1
    yscale = (KLratio^-capshare)./(ygrid'*ydist);
    
    % initialize consumption function in first iteration only
    if iterKL==1
        conguess = zeros(na,ny);
        for iy = 1:ny
            conguess(:,iy) = r.*agrid + wage.* yscale.*ygrid(iy);
        end
        con = conguess;
     end

    % solve for policy functions with EGP
    
    iter = 0;
    cdiff = 1000;
    while iter <= max_iter && cdiff>tol_iter
        iter = iter + 1;
        sav = zeros(na,ny);

        conlast = con;

        emuc = u1(conlast)*ydist; 
        muc1 = beta.*R.*emuc; 
        con1 = u1inv(muc1);

        % loop over income
        for iy = 1:ny

            ass1(:,iy) = (con1 + agrid - wage.* yscale.*ygrid(iy))./R;

            % loop over current period ssets
            for ia  = 1:na 
                if agrid(ia)<ass1(1,iy) %borrowing constraint binds
                    sav(ia,iy) = borrow_lim;
                else %borrowing constraint does not bind;
                    sav(ia,iy) = lininterp1(ass1(:,iy),agrid,agrid(ia));
                end                
            end
            con(:,iy) = R.*agrid +wage.* yscale.*ygrid(iy) - sav(:,iy);
        end   

        cdiff = max(max(abs(con-conlast)));
        if Display >=2
            disp([' Iteration no. ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
        end
    end    


    %simulate: start at assets from last interation
    if iterKL==1
        asim = zeros(Nsim,Tsim);
    elseif iterKL>1
%         asim(:,1) = Ea.*ones(Nsim,1);
        asim(:,1) = asim(:,Tsim);
    end
    
    %create interpolating function
    for iy = 1:ny
        savinterp{iy} = griddedInterpolant(agrid,sav(:,iy),'linear');
    end
    
    %loop over time periods
    for it = 1:Tsim
        if Display >=2 && mod(it,100) ==0
            disp([' Simulating, time period ' int2str(it)]);
        end
                
        % asset choice
        if it<Tsim
            for iy = 1:ny
                asim(yindsim(:,it)==iy,it+1) = savinterp{iy}(asim(yindsim(:,it)==iy,it));
            end
        end
    end
    
    %assign actual labor income values;
    labincsim = wage.*yscale.*ysim;

    % mean assets and efficiency units
    Ea  = mean(asim(:,Tsim));
    L   = yscale.*mean(ysim(:,Tsim));
    
    KLrationew = Ea./ L;
    
    KLdiff = KLrationew./KLratio - 1;
    if Display >=1
        disp(['Equm iter ' int2str(iterKL), ', r  = ',num2str(r), ', KL ratio: ',num2str(KLrationew),' KL diff: ' num2str(KLdiff*100) '%']);
    end

    KLratio = (1-step_KL)*KLratio + step_KL*KLrationew; 
end

 
%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption policy function
    subplot(2,4,1);
    plot(agrid,con(:,1),'b-',agrid,con(:,ny),'r-','LineWidth',1);
    grid;
    xlim([0 amax]);
    title('Consumption Policy Function');
    legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(agrid,sav(:,1)-agrid,'b-',agrid,sav(:,ny)-agrid,'r-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 amax]);
    title('Savings Policy Function (a''-a)');
    
    % consumption policy function: zoomed in
    subplot(2,4,3);
    plot(agrid,con(:,1),'b-o',agrid,con(:,ny),'r-o','LineWidth',2);
    grid;
    xlim([0 1]);
    title('Consumption: Zoomed');
    
     % savings policy function: zoomed in
    subplot(2,4,4);
    plot(agrid,sav(:,1)-agrid,'b-o',agrid,sav(:,ny)-agrid,'r-o','LineWidth',2);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 1]);
    title('Savings: Zoomed (a''-a)');
    
    
    %income distribution
    subplot(2,4,5);
    hist(ysim(:,Tsim),ygrid);
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
    ylim([0 2*Ea]);
    ylabel('Time Period');
    title('Mean Asset Convergence');
    
    
   % asset distribution statistics
    aysim = asim(:,Tsim) ./ mean(ysim(:,Tsim));
    disp(['Mean assets: ' num2str(mean(aysim))]);
    disp(['Fraction borrowing constrained: ' num2str(sum(aysim==borrow_lim)./Nsim * 100) '%']);
    disp(['10th Percentile: ' num2str(quantile(aysim,.1))]);
    disp(['50th Percentile: ' num2str(quantile(aysim,.5))]);
    disp(['90th Percentile: ' num2str(quantile(aysim,.9))]);
    disp(['99th Percentile: ' num2str(quantile(aysim,.99))]);

end
