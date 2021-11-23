% Value Function Iteration with IID Income
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS

% preferences
risk_aver   = 2;
beta        = 0.95;

%returns
r           = 0.03;
%r = 1/beta - 1;
R = 1+ r;

% income risk: discretized N(mu,sigma^2)
mu_y    = 1;
sd_y    = 0.2;
ny      = 5;

% asset grids
na          = 500;
amax        = 20; 
borrow_lim  = 0;
agrid_par   = 1; %1 for linear, 0 for L-shaped

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;


%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

%% DRAW RANDOM NUMBERS
rng(2017);
yrand = rand(Nsim,Tsim);

%% SET UP GRIDS

% assets
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

% income: disretize normal distribution
width = fzero(@(x)discrete_normal(ny,mu_y,sd_y,x),2);
[temp,ygrid,ydist] = discrete_normal(ny,mu_y,sd_y,width);
ycumdist = cumsum(ydist);


%% UTILITY FUNCTION

if risk_aver==1
    u = @(c)log(c);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
end    

%% INITIALIZE VALUE FUNCTION

Vguess = zeros(na,ny);
for iy = 1:ny
    Vguess(:,iy) = u(r.*agrid+ygrid(iy))./(1-beta);
end
% Vguess = ones(na,ny);

%% ITERATE ON VALUE FUNCTION

V = Vguess;

Vdiff = 1;
iter = 0;

while iter <= max_iter && Vdiff>tol_iter
    iter = iter + 1;
    Vlast = V;
    V = zeros(na,ny);
    sav = zeros(na,ny);
    savind = zeros(na,ny);
    con = zeros(na,ny);
    
    % loop over assets
    for ia = 1:na
        
        % loop over income
        for iy = 1:ny
            cash = R.*agrid(ia) + ygrid(iy);
            Vchoice = u(max(cash-agrid,1.0e-10)) + beta.*(Vlast*ydist);           
            [V(ia,iy),savind(ia,iy)] = max(Vchoice);
            sav(ia,iy) = agrid(savind(ia,iy));
            con(ia,iy) = cash - sav(ia,iy);
       end
    end
    
    
    Vdiff = max(max(abs(V-Vlast)));
    if Display >=1
        disp(['Iteration no. ' int2str(iter), ' max val fn diff is ' num2str(Vdiff)]);
    end
end    



%% SIMULATE
if DoSimulate ==1
    yindsim = zeros(Nsim,Tsim);
    aindsim = zeros(Nsim,Tsim);
    
    % initial assets
    aindsim(:,1) = 1;
    
    %loop over time periods
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
            disp([' Simulating, time period ' int2str(it)]);
        end
        
        %income realization: note we vectorize simulations at once because
        %of matlab, in other languages we would loop over individuals
        yindsim(yrand(:,it)<= ycumdist(1),it) = 1;
        for iy = 2:ny
            yindsim(yrand(:,it)> ycumdist(iy-1) & yrand(:,it)<=ycumdist(iy),it) = iy;
        end
        
        % asset choice
        if it<Tsim
            for iy = 1:ny
                aindsim(yindsim(:,it)==iy,it+1) = savind(aindsim(yindsim(:,it)==iy,it),iy);
            end
        end
    end
    
    %assign actual asset and income values;
    asim = agrid(aindsim);
    ysim = ygrid(yindsim);

end

%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption policy function
    subplot(2,4,1);
    plot(agrid,con(:,1),'b-',agrid,con(:,ny),'r-','LineWidth',1);
    grid;
    xlim([0 amax]);
%     title('Consumption Policy Function');
    title('Consumption');
    legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(agrid,sav(:,1)-agrid,'b-',agrid,sav(:,ny)-agrid,'r-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 amax]);
%     title('Savings Policy Function (a''-a)');
    title('Savings');
    
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
    hist(asim(:,Tsim),[0:0.05:2]);
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
    disp(['Mean assets: ' num2str(mean(aysim))]);
    disp(['Fraction borrowing constrained: ' num2str(sum(aysim==borrow_lim)./Nsim * 100) '%']);
    disp(['10th Percentile: ' num2str(quantile(aysim,.1))]);
    disp(['50th Percentile: ' num2str(quantile(aysim,.5))]);
    disp(['90th Percentile: ' num2str(quantile(aysim,.9))]);
    disp(['99th Percentile: ' num2str(quantile(aysim,.99))]);

end
