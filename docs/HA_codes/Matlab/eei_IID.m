% Euler Equation Iteration with IID Income
% Greg Kaplan 2017

clear;
close all;

%% GLOBALS
global agrid conlast u1 beta ny R ydist cash;

%% PARAMETERS

% preferences
risk_aver   = 2;
beta        = 0.95;

%returns
r           = 0.03;
R = 1+ r;

% income risk: discretized N(mu,sigma^2)
mu_y    = 1;
sd_y    = 0.2;
ny      = 5;

% asset grids
na          = 30;
amax        = 30; 
borrow_lim  = 0;
agrid_par   = 0.4; %1 for linear, 0 for L-shaped

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;

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

u1 = @(c) c.^(-risk_aver);

%% INITIALIZE CONSUMPTION FUNCTION

conguess = zeros(na,ny);
for iy = 1:ny
    conguess(:,iy) = r.*agrid+ygrid(iy);
end


%% ITERATE ON EULER EQUATION

con = conguess;
emuc = u1(con)*ydist; 

iter = 0;
cdiff = 1000;

while iter <= max_iter && cdiff>tol_iter
    iter = iter + 1;
    conlast = con;
    sav = zeros(na,ny);
    
    % loop over assets
    for ia = 1:na
        
        % loop over income
        for iy = 1:ny
            cash = R.*agrid(ia) + ygrid(iy);
            
            %use consumption interpolation
            if InterpCon==1
                if fn_eeqn_c(borrow_lim)>=0 %check if borrowing constrained
                    sav(ia,iy) = borrow_lim;
                else
                    sav(ia,iy) = fzero(@(x)fn_eeqn_c(x),0.5*cash,options);
                end    
                
            %use expected marginal utility interpolation
            elseif InterpEMUC==1
                if u1(cash-borrow_lim) >= beta.*R.*lininterp1(agrid,emuc,borrow_lim) %check if borrowing constrained
                    sav(ia,iy) = borrow_lim;
                else
                    sav(ia,iy) = fzero(@(x)u1(cash-x)-beta.*R.*lininterp1(agrid,emuc,x), 0.5*cash, options);                
                end
                
            end    
           con(ia,iy) = cash - sav(ia,iy);
       end
    end
    
    emuc = u1(con)*ydist; 

    cdiff = max(max(abs(con-conlast)));
    if Display >= 1
        disp(['Iteration no. ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
    end
end    


%% SIMULATE
if DoSimulate==1

    yindsim = zeros(Nsim,Tsim);
    asim = zeros(Nsim,Tsim);
    
    %create interpolating function
    for iy = 1:ny
        savinterp{iy} = griddedInterpolant(agrid,sav(:,iy),'linear');
    end
    
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
                asim(yindsim(:,it)==iy,it+1) = savinterp{iy}(asim(yindsim(:,it)==iy,it));
            end
        end
    end
    
    %assign actual income values;
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
