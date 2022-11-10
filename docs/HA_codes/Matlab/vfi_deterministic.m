% Deterministic Value Function Iteration
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS

% preferences
risk_aver   = 2;
beta        = 0.95;

%returns
r           = 0.03;
R = 1+ r;

%income
y = 1;

% asset grids
na          = 1000;
amax        = 20; 
borrow_lim  = 0;
agrid_par   = 1; %1 for linear, 0 for L-shaped

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 100;
Tsim        = 500;


%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

%% SET UP GRIDS

% assets
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

%% DRAW RANDOM NUMBERS
rng(2017);
arand = rand(Nsim,1);


%% UTILITY FUNCTION

if risk_aver==1
    u = @(c)log(c);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
end    

u1 = @(c) c.^(-risk_aver);

%% INITIALIZE VALUE FUNCTION

Vguess = u(r.*agrid+y)./(1-beta);

%% ITERATE ON VALUE FUNCTION

V = Vguess;

Vdiff = 1;
iter = 0;

while iter <= max_iter && Vdiff>tol_iter
    iter = iter + 1;
    Vlast = V;
    V = zeros(na,1);
    sav = zeros(na,1);
    savind = zeros(na,1);
    con = zeros(na,1);
    
    % loop over assets
    for ia = 1:na
        
        cash = R.*agrid(ia) + y;
        Vchoice = u(max(cash-agrid,1.0e-10)) + beta.*Vlast;           
        [V(ia),savind(ia)] = max(Vchoice);
        sav(ia) = agrid(savind(ia));
        con(ia) = cash - sav(ia);
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
    
    % initial assets: uniform on [borrow_lim, amax]    
    ainitial = borrow_lim + arand.*(amax-borrow_lim);
    
    %allocate to nearest point on agrid;
    aindsim(:,1) = interp1(agrid,[1:na]',ainitial,'nearest');
        
    %loop over time periods
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
            disp([' Simulating, time period ' int2str(it)]);
        end
        % asset choice
        if it<Tsim
            aindsim(:,it+1) = savind(aindsim(:,it));
        end
    end
    
    %assign actual asset and income values;
    asim = agrid(aindsim);
    csim = R.*asim(:,1:Tsim-1) + y - asim(:,2:Tsim); 
end

%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption policy function
    subplot(2,4,1);
    plot(agrid,con,'b-','LineWidth',1);
    grid;
    xlim([0 amax]);
    title('Consumption Policy Function');
%     legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(agrid,sav-agrid,'b-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 amax]);
    title('Savings Policy Function (a''-a)');
    
    % consumption policy function: zoomed in
    subplot(2,4,3);
    plot(agrid,con,'b-o','LineWidth',2);
    grid;
    xlim([0 1]);
    title('Consumption: Zoomed');
    
     % savings policy function: zoomed in
    subplot(2,4,4);
    plot(agrid,sav-agrid,'b-o','LineWidth',2);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 1]);
    title('Savings: Zoomed (a''-a)');
    
    
    %asset dynamics distribution
    subplot(2,4,5:6);
    plot([1:Tsim]',asim);
    grid on;
    title('Asset Dynamics');

    %consumption dynamics distribution
    subplot(2,4,7:8);
    plot([1:Tsim-1]',csim);
    grid on;
    title('Consumption Dynamics');

end
