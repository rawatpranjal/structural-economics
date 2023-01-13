% Deterministic Value Function Iteration
% Greg Kaplan 2017

clear;
close all;

%% PARAMETERS

%horizon
T = 50;

% preferences
risk_aver   = 2;
beta        = 0.97;

%returns
r           = 0.05;
R = 1+ r;

%income
y = zeros(T,1);
y(1:30) = [1:30]';
y(31:50) = 5;

%rescale income so that average income = 1;
y = y./sum(y);

% asset grids
na          = 1000;
amax        = 5; 
% borrow_lim  = 0;
borrow_lim  = -0.10;
agrid_par   = 1; %1 for linear, 0 for L-shaped


%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

%% SET UP GRIDS

% assets
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

%put explicit point at a=0
agrid(agrid==min(abs(agrid-0))) = 0;

%% UTILITY FUNCTION

if risk_aver==1
    u = @(c)log(c);
else    
    u = @(c)(c.^(1-risk_aver)-1)./(1-risk_aver);
end    

u1 = @(c) c.^(-risk_aver);

%% INITIALIZE ARRAYS
V = zeros(na,T);
con = zeros(na,T);
sav = zeros(na,T);
savind = zeros(na,T);

%% DECISIONS AT t=T
savind(:,T) = find(agrid==0);
sav(:,T) = 0;
con(:,T) = R.*agrid + y(T) - sav(:,T);
V(:,T) = u(con(:,T));


%% SOlVE VALUE FUNCTION BACKWARD 

for it = T-1:-1:1
    if Display >=1 
        disp(['Solving at age: ' int2str(it)]);
    end
    
    
    % loop over assets
    for ia = 1:na
        
        cash = R.*agrid(ia) + y(it);
        Vchoice = u(max(cash-agrid,1.0e-10)) + beta.*V(:,it+1);           
        [V(ia,it),savind(ia,it)] = max(Vchoice);
        sav(ia,it) = agrid(savind(ia,it));
        con(ia,it) = cash - sav(ia,it);
    end
    
end    


%% SIMULATE
if DoSimulate ==1
    aindsim = zeros(T+1,1);
    
    % initial assets: uniform on [borrow_lim, amax]    
    ainitial = 0;
    
    %allocate to nearest point on agrid;
    aindsim(1) = interp1(agrid,[1:na]',ainitial,'nearest');
        
    %simulate forward
    for it = 1:T
        disp([' Simulating, time period ' int2str(it)]);

        % asset choice
        aindsim(it+1) = savind(aindsim(it),it);
    end
    
    %assign actual asset and income values;
    asim = agrid(aindsim);
    csim = R.*asim(1:T) + y - asim(2:T+1); 
end


%% MAKE PLOTS
if MakePlots ==1 
    figure(1);
    
    % consumption and income path
    subplot(1,2,1);
    plot([1:50]',y,'k-',[1:50]',csim,'r--','LineWidth',1);
    grid;
    title('Income and Consumption');
    legend('Income','Consumpion');

    % wealth path function
    subplot(1,2,2);
    plot([0:50]',asim,'b-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    title('Wealth');
        

end
