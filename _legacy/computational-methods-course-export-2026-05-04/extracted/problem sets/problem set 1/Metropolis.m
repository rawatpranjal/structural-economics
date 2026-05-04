%==========================================================================
%               METROPOLIS HASTINGS ALGORITHM (Example from Lecture 1)
%==========================================================================

tic
close all
clear
clc


%=========================================================================
%                       SETTING THE PARAMETERS
%=========================================================================


Mu1    = [1.5,1.5];       % Mean of the First Density

Mu2    = [-1.5,-1.5];     % Mean of Second Density

Sigma  = [1,0.5;0.5,1];   % Variance-Covariance Matrix

p      = 0.5;             % Mixing probability;

Nsim   = 5000;            % Number of Draws for M-H algorithm

Nburn  = int32(0.05*Nsim); % Number of Draws to discart

Theta  = zeros(Nsim,2);   % Vector Collecting Draws for Theta;

Sigcand    = eye(2);      % Variance-Covariance Matrix of Candidate Density

scale      = 0.5^(2);     % Scale for Candidate Density

Theta(1,:) = [10,-10];    % Initial Value for the Gibbs Sampler

liki = log(p*mvnpdf(Theta(1,:),Mu1,Sigma) + (1-p)*mvnpdf(Theta(1,:),Mu2,Sigma));

accept = 0;

%=========================================================================
%                  METROPOLIS HASTINGS ALGORITHM
%=========================================================================

for i=2:Nsim

    % Draw from the candidate density

    Thetac  = mvnrnd(Theta(i-1,:),scale*Sigcand);

    % Evaluate the Likelihood function at the candidate draw

    likic  = log(p*mvnpdf(Thetac,Mu1,Sigma) + (1-p)*mvnpdf(Thetac,Mu2,Sigma));

    % Calculate Acceptance Probability

    a      = min(1,exp(likic-liki));

    u      = rand(1);

    if u<=a
        Theta(i,:) = Thetac;
        liki       = likic;
        accept     = accept+1;
        % Rate of Acceptance of Candidate Draws
    acceptancerate = accept/i;
    else
        Theta(i,:) = Theta(i-1,:);
    end

end

%=========================================================================
%                   THE METROPOLIS-HASTINGS "WALK"
%=========================================================================

% Generating the Likelihood Contours

u1  = max(abs(Theta(:,1)));
u2  = max(abs(Theta(:,2)));

Th1 = linspace(-u1,u1,50);
Th2 = linspace(-u2,u2,50);
Z   = zeros(50,50);

for j=1:50
    for i=1:50
      Z(i,j)= log(p*mvnpdf([Th1(j),Th2(i)],Mu1,Sigma) + ...
              (1-p)*mvnpdf([Th1(j),Th2(i)],Mu2,Sigma));
    end
end

figure('Position',[20,20,900,600],'Name','The Metropolis-Hastings "Walk"','Color','w')

scatter(Theta(:,1),Theta(:,2),12), hold on
[C]= contour(Th1,Th2,Z,8,'LineWidth',1.5);
xlim([-u2-1,u2+1])
ylim([-u1-1,u1+1])
xlabel('\theta_{1}','FontSize',18,'FontWeight','bold')
ylabel('\theta_{2}','FontSize',18,'FontWeight','bold')
clabel(C,'FontSize',13,'FontWeight','bold')
title('The Metropolis-Hastings Walk','FontSize',13,'FontWeight','bold');


%=========================================================================
%                         PLOTTING THE DRAWS
%=========================================================================


pnames = strvcat( '\theta_{1} Draws','\theta_{2} Draws');

figure('Position',[20,20,900,600],'Name','Metropolis-Hastings Draws','Color','w')

trueval(:,1) = (p*Mu1(1)+(1-p)*Mu2(1))*ones(Nsim,1);    % True Value for the mean  of Theta1

trueval(:,2) = (p*Mu1(2)+(1-p)*Mu2(2))*ones(Nsim,1);

for i=1:2
subplot(2,1,i), plot(Theta(:,i),'LineStyle','-','Color','b',...
        'LineWidth',2.5), hold on
    plot(trueval(:,i),'LineStyle',':','Color','black',...
        'LineWidth',2.5);
xlabel('Number of Draws','FontWeight','bold')
title(pnames(i,:),'FontSize',13,'FontWeight','bold');
end


%=========================================================================
%                         CUMULATIVE MEANS
%=========================================================================
Theta      = Theta(Nburn:end,:);
[Nsim,o]   = size(Theta);

rmean = zeros(Nsim,2);     % Vector Collecting Cumulative Means
rvar1 = zeros(Nsim,1);     % Vector Collecting Cumulative Variance of Theta1
rcov  = zeros(Nsim,1);     % Vector Collecting Cumulative Covariance of draws

for i=1:Nsim

    rmean(i,:) = mean([Theta(1:i,1),Theta(1:i,2)]);
    A          = cov(Theta(1:i,1),Theta(1:i,2));
    rvar1(i,:) = A(2,2);
    rcov (i,:) = A(1,2);

end

Stat           = [rmean,rvar1,rcov];

truevalue(:,1) = (p*Mu1(1)+(1-p)*Mu2(1))*ones(Nsim,1);

truevalue(:,2) = (p*Mu1(2)+(1-p)*Mu2(2))*ones(Nsim,1);

truevalue(:,3) = Sigma(1,1)*ones(Nsim,1);

truevalue(:,4) = Sigma(1,2)*ones(Nsim,1);


pnames = strvcat( 'Mean(\theta_{1})','Mean(\theta_{2})',...
                  'Var(\theta_{1})','Cov(\theta_{1},\theta_{2})');

figure('Position',[20,20,900,600],'Name','Cumulative Means','Color','w')

for i=1:2
subplot(1,2,i), plot(Stat(:,i),'LineStyle','-','Color','b',...
        'LineWidth',2.5), hold on
    plot(truevalue(:,i),'LineStyle',':','Color','black',...
        'LineWidth',2.5);
xlabel('Number of Draws','FontWeight','bold')
title(pnames(i,:),'FontSize',13,'FontWeight','bold');
end

%=========================================================================
%                     AUTOCORRELATION OF DRAWS
%=========================================================================

ACmax     = int16(0.10*Nsim);

Thetadev  = [Theta(:,1)-mean(Theta(:,1)),Theta(:,2)-mean(Theta(:,2))];

Thetadev2 = [Theta(:,1).^(2) - mean(Theta(:,1).^(2)),Theta(:,2).^(2) - ...
            mean(Theta(:,2).^(2))];

tdev      = [Thetadev,Thetadev2];

Autocorr  = zeros(ACmax,4);

for j=1:4
    for i=1:ACmax

   Autocorr(i,j) = ((tdev(1:Nsim-i,j)'*tdev(1:Nsim-i,j))^(-1))*...
                   tdev(1:Nsim-i,j)'*tdev(1+i:Nsim,j);

    end
end

pnames = strvcat( '\theta_{1}','\theta_{2}',...
                  '\theta^{2}_{1}','\theta^{2}_{2}');

figure('Position',[20,20,900,600],'Name','Autocorrelation','Color','w')

for i=1:2
subplot(2,1,i), plot(Autocorr(:,i),'LineStyle','-','Color','b',...
        'LineWidth',2.5), hold on
xlabel('Order','FontWeight','bold')
ylabel('Autocorrelation','FontWeight','bold')
title(pnames(i,:),'FontSize',13,'FontWeight','bold');
end

elapsedtime=toc;