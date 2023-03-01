% plot_posterior.m plots the posterior probabilities for drawing from Bingo cage A as a funtion of prior for all outcomes (0,...,6)
%                  based on 1995 JASA paper of El-Gamal and Grether, "Are People Bayesian? Uncovering Behavioral Strategies"

% temporarily fixed parameters relevant for the California experiments. Later adjust these for values relevant to Wisconsin experiments

pa=2/3;  % probability of drawing a ball marked N from bingo cage A (4 N balls, 2 G balls)
pb=1/2;  % probability of drawing a ball marked N from bingo cage B (3 N balls, 3 G balls)
nballs=6;  % number of balls in each bingo cage (draws are done with replacement)

% these parameters you can change and play with to understand the model and estimation

dependent_variable='true_classification';

sample_type='all_pooled_california';  % choices: 'all_pooled' or 'all_individual'
model='llr_lpr';  % possible model types currently, prior_linearly  prior_as_log_odds and llr_lpr

training_sample_size=10;

% first estimate the parameters of the model of subjective posterior beliefs using the human subject data 

load datastruct;
truetheta=[.1 0 -1 -1]';
[ydata,xdata]=learning_how_to_learn.prepare_data(datastruct,sample_type);

   options=optimoptions('fminunc','FunctionTolerance',1e-10,'Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
   fprintf('Estimating structural binary logit model of subject choices using model specification for subjective posterior: %s\n',model);
   tic;
   [structural_thetahat,sllf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.structural_blogit(ydata,xdata,theta),truetheta,options);
   save('structural_thetahat','structural_thetahat');
   sllf=-sllf;
   fprintf('maximum likelihood estimates  standard errors gradient of loglikelihood\n');
   names=cell(4,1);
   names{1}='sigma';
   names{2}='constant';
   names{3}='log likelihood ratio';
   names{4}='log prior ratio';
   stderr=sqrt(diag(inv(hessian)));
   for i=1:4
     fprintf('%20s %10.4f %10.4f %10.4e\n',names{i},structural_thetahat(i),stderr(i),grad(i));
   end
   fprintf('\nMaximum likelihood estimation of a restricted "noisy Bayesian" model\n');
   [restricted_sigma,llfr]=fminbnd(@(x) learning_how_to_learn.structural_blogit(ydata,xdata,[x; truetheta(2:4)]),0,1);
   llfr=-llfr;
   fprintf('Maximum likelihood estimate of sigma in restricted "noisy Bayesian model" model %g, log-likehood value: %g\n',restricted_sigma,llfr);
   fprintf('P-value of likeihood ratio test of noisy Bayesian model vs unrestricted structural logit model: %g\n',chi2cdf(2*(sllf-llfr),3,'upper'));
   toc

% now for comparison, train a machine learning algorithm (2 level neural network) using correct identifications of which urn the balls
% were drawn from in a training sample of size training_sample_size. This provides a base case to compare to human
% decision makers in the experiment, who were "untrained".

[training_ydata,training_xdata]=learning_how_to_learn.generate_training_data(training_sample_size,dependent_variable);
tic;
[trainedthetahat,llf,exitflag,output,grad,hessian]=...
fminunc( @(theta) learning_how_to_learn.structural_blogit(training_ydata,training_xdata,theta),truetheta,options);
toc

if (exitflag)
fprintf('training of learning machine has completed using %i training instances (i.e. observations)\n',training_sample_size);
truetheta=[0 0 -1 -1]';
fprintf('converged log-likelihood value %g  log-likelihood at true parameters %g\n',llf,...
 learning_how_to_learn.structural_blogit(training_ydata,training_xdata,truetheta));
else
  fprintf('fminunc terminated with exitflag %i, algorithm may not have converged\n',exitflag);
end

%trained_stderr=sqrt(diag(inv(hessian)/training_sample_size));

fprintf('\nEstimated coefficients by maximum likelihood using the likelihood function in learning_how_to_learn.logit and Matlab fminunc command\n');
fprintf('Estimated and true parameter vectors and std errors (last column)\n');
[trainedthetahat truetheta]


% Rest of the code does some calculations and plots some graphs

binprobs_a=binopdf((0:nballs),nballs,pa);
binprobs_b=binopdf((0:nballs),nballs,pb);

prior=(0:.01:1)';
nprior=size(prior,1);

posterior_a=zeros(nprior,nballs+1);  % posterior probability for outcome with 0 N balls in sample as a function of prior

for i=1:nprior;

   posterior_a(i,:)=binprobs_a*prior(i);
   posterior_a(i,:)=posterior_a(i,:)./(posterior_a(i,:)+binprobs_b*(1-prior(i)));

end

trained_posterior_a=zeros(size(prior,1),nballs+1);
human_posterior_a=zeros(size(prior,1),nballs+1);
trained_prob_a=zeros(size(prior,1),nballs+1);
for i=0:nballs;
    trained_posterior_a(:,i+1)=learning_how_to_learn.subjective_posterior_prob([i*ones(101,1) prior],trainedthetahat(2:4),model);
    human_posterior_a(:,i+1)=learning_how_to_learn.subjective_posterior_prob([i*ones(101,1) prior],structural_thetahat(2:4),model);
    trained_prob_a(:,i+1)=learning_how_to_learn.structural_ccp([i*ones(101,1) prior],trainedthetahat);
    human_prob_a(:,i+1)=learning_how_to_learn.structural_ccp([i*ones(101,1) prior],structural_thetahat);
end;

posteriors=sort([posterior_a(:) human_posterior_a(:) trained_posterior_a(:) trained_prob_a(:) human_prob_a(:)],1);



figure(1);
clf
hold on;
plot(prior,posterior_a,'Linewidth',2);
plot(prior,.5*ones(nprior,1),'k:','Linewidth',2);
title('True posterior probabilities of drawing outcomes, by prior probability');
ylabel(sprintf('Posterior probability P(A|n,\\pi_A)'));
xlabel(sprintf('Prior probability \\pi_A'));
legend(sprintf('P(A|0,\\pi_A)'),sprintf('P(A|1,\\pi_A)'),sprintf('P(A|2,\\pi_A)'),sprintf('P(A|3,\\pi_A)'),...
sprintf('P(A|4,\\pi_A)'),sprintf('P(A|5,\\pi_A)'),sprintf('P(A|6,\\pi_A)'),'Location','Northwest');
axis square;
hold off;

figure(2);
clf
hold on;
plot(prior,trained_posterior_a,'Linewidth',2);
plot(prior,.5*ones(nprior,1),'k:','Linewidth',2);
title('Trained posterior probabilities of drawing outcomes, by prior probability');
ylabel(sprintf('Trained posterior probability P(A|n,\\pi_A)'));
xlabel(sprintf('Prior probability \\pi_A'));
legend(sprintf('P(A|0,\\pi_A)'),sprintf('P(A|1,\\pi_A)'),sprintf('P(A|2,\\pi_A)'),sprintf('P(A|3,\\pi_A)'),...
sprintf('P(A|4,\\pi_A)'),sprintf('P(A|5,\\pi_A)'),sprintf('P(A|6,\\pi_A)'),'Location','Northwest');
axis square;
hold off;

figure(3);
clf
hold on;
plot(prior,human_posterior_a,'Linewidth',2);
plot(prior,.5*ones(nprior,1),'k:','Linewidth',2);
title('Estimated subjective posterior beliefs of human subjects, by prior probability');
ylabel(sprintf('Estimated subjective posterior probability P(A|n,\\pi_A)'));
xlabel(sprintf('Prior probability \\pi_A'));
legend(sprintf('P(A|0,\\pi_A)'),sprintf('P(A|1,\\pi_A)'),sprintf('P(A|2,\\pi_A)'),sprintf('P(A|3,\\pi_A)'),...
sprintf('P(A|4,\\pi_A)'),sprintf('P(A|5,\\pi_A)'),sprintf('P(A|6,\\pi_A)'),'Location','Northwest');
axis square;
hold off;



figure(4);
clf
hold on;
plot(posteriors(:,1),posteriors(:,2),'r-','Linewidth',2);
plot(posteriors(:,1),posteriors(:,3),'b-','Linewidth',2);
plot(posteriors(:,1),posteriors(:,1),'k-','Linewidth',2);
plot(posteriors(:,1),.5*ones(size(posteriors,1),1),'k-','Linewidth',1);
legend('Estimated human subject posterior','Trained machine posterior','45 degree line, true posterior','Location','Northwest');
title('Estimated subjective posteriors of human subjects vs trained machine posterior');
xlabel(sprintf('True posterior probability P(A|n,\\pi_A)'));
axis square;
hold off;


true_cutoffs=learning_how_to_learn.subjective_cutoffs(truetheta(2:4));
implied_human_cutoffs=learning_how_to_learn.subjective_cutoffs(structural_thetahat(2:4));
implied_machine_cutoffs=learning_how_to_learn.subjective_cutoffs(structural_thetahat(2:4));

figure(5);
clf
hold on;
line([0 true_cutoffs(7)],[6 6],'Linewidth',2,'Color','k');
line([true_cutoffs(7) true_cutoffs(7)],[6 5],'Linewidth',2,'Color','k');
line([true_cutoffs(7) true_cutoffs(6)],[5 5],'Linewidth',2,'Color','k');
line([true_cutoffs(6) true_cutoffs(6)],[5 4],'Linewidth',2,'Color','k');
line([true_cutoffs(6) true_cutoffs(5)],[4 4],'Linewidth',2,'Color','k');
line([true_cutoffs(5) true_cutoffs(5)],[4 3],'Linewidth',2,'Color','k');
line([true_cutoffs(5) true_cutoffs(4)],[3 3],'Linewidth',2,'Color','k');
line([true_cutoffs(4) true_cutoffs(4)],[3 2],'Linewidth',2,'Color','k');
line([true_cutoffs(4) true_cutoffs(3)],[2 2],'Linewidth',2,'Color','k');
line([true_cutoffs(3) true_cutoffs(3)],[2 1],'Linewidth',2,'Color','k');
line([true_cutoffs(3) true_cutoffs(2)],[1 1],'Linewidth',2,'Color','k');
line([true_cutoffs(2) true_cutoffs(2)],[1 0],'Linewidth',2,'Color','k');
line([true_cutoffs(2) true_cutoffs(1)],[0 0],'Linewidth',2,'Color','k');
line([true_cutoffs(1) true_cutoffs(1)],[0 -1],'Linewidth',2,'Color','k');
line([true_cutoffs(1) 1],[-1 -1],'Linewidth',2,'Color','k');

text(.6,4,'Choose A');
text(.1,2,'Choose B');
title('Decision regions implied by Bayes Rule');
xlabel(sprintf('Prior probability \\pi_A'));
ylabel(sprintf('Number of balls marked N, n'));
hold off;


figure(6);
clf
hold on;
line([0 true_cutoffs(7)],[6 6],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(7)],[6 5],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(6)],[5 5],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(6)],[5 4],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(5)],[4 4],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(5)],[4 3],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(4)],[3 3],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(4)],[3 2],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(3)],[2 2],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(3)],[2 1],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(2)],[1 1],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(2)],[1 0],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(1)],[0 0],'Linewidth',1,'Color','k');
line([true_cutoffs(1) true_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k');
line([true_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k');

line([0 implied_machine_cutoffs(7)],[6 6],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(7) implied_machine_cutoffs(7)],[6 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(7) implied_machine_cutoffs(6)],[5 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(6) implied_machine_cutoffs(6)],[5 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(6) implied_machine_cutoffs(5)],[4 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(5) implied_machine_cutoffs(5)],[4 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(5) implied_machine_cutoffs(4)],[3 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(4) implied_machine_cutoffs(4)],[3 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(4) implied_machine_cutoffs(3)],[2 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(3) implied_machine_cutoffs(3)],[2 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(3) implied_machine_cutoffs(2)],[1 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(2) implied_machine_cutoffs(2)],[1 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(2) implied_machine_cutoffs(1)],[0 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(1) implied_machine_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_machine_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k','LineStyle',':');

text(.88,5.7,'Choose A');
text(.05,-.5,'Choose B');

b=find(training_ydata==0);
bpriors=training_xdata(b,2);
bdraws=training_xdata(b,1);
scatter(bpriors,bdraws,25,'MarkerEdgeColor','b','MarkerFaceColor','b');
a=find(training_ydata==1);
apriors=training_xdata(a,2);
adraws=training_xdata(a,1);
scatter(apriors,adraws,25,'MarkerEdgeColor','r','MarkerFaceColor','r');

title({'Decision regions implied by Bayes Rule, with training observations added'},...
   {'Solid line plots true cutoffs for choosing A vs B, dotted line plots cutoffs for trained machine'});
xlabel(sprintf('Prior probability \\pi_A'));
ylabel(sprintf('Number of balls marked N, n'));
hold off;


figure(7);
clf
hold on;
line([0 true_cutoffs(7)],[6 6],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(7)],[6 5],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(6)],[5 5],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(6)],[5 4],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(5)],[4 4],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(5)],[4 3],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(4)],[3 3],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(4)],[3 2],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(3)],[2 2],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(3)],[2 1],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(2)],[1 1],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(2)],[1 0],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(1)],[0 0],'Linewidth',1,'Color','k');
line([true_cutoffs(1) true_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k');
line([true_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k');

line([0 implied_human_cutoffs(7)],[6 6],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(7) implied_human_cutoffs(7)],[6 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(7) implied_human_cutoffs(6)],[5 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(6) implied_human_cutoffs(6)],[5 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(6) implied_human_cutoffs(5)],[4 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(5) implied_human_cutoffs(5)],[4 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(5) implied_human_cutoffs(4)],[3 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(4) implied_human_cutoffs(4)],[3 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(4) implied_human_cutoffs(3)],[2 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(3) implied_human_cutoffs(3)],[2 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(3) implied_human_cutoffs(2)],[1 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(2) implied_human_cutoffs(2)],[1 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(2) implied_human_cutoffs(1)],[0 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(1) implied_human_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k','LineStyle',':');

text(.88,5.7,'Choose A');
text(.01,-.5,'Choose B');
text(.2,-.25,'Size of circles proportional to number of observations');
text(.2,-.5,'Shading of circles indicates fraction choosing A or B');
text(.2,-.75,'Light blue is 100% B, light red is 100% A, purple if 50% A, etc');

red=[1,0,0];
blue=[0,0,1];
ndraws=(0:nballs)';
npriors=[1/3; 1/2; 2/3];
total_nobs=0;
for nb=1:nballs+1
 for np=1:3
    frac=mean(ydata(find(xdata(:,1)==ndraws(nb) & abs(xdata(:,2)-npriors(np))<.01)));
    nobs=sum(xdata(:,1)==ndraws(nb) & abs(xdata(:,2)-npriors(np))<.01);
    total_nobs=total_nobs+nobs;
    if (nobs)
    color=frac*red+(1-frac)*blue;
    %fprintf('balls %i prior %g frac %g nobs=%i\n',ndraws(nb),npriors(np),frac,nobs);
    ms=1+40*(nobs/600);
    plot(npriors(np),ndraws(nb),'o','Markersize',ms,'MarkerEdgeColor',color,'MarkerFaceColor',color);
    end
 end
end

title({'Decision regions implied by Bayes Rule, with subject choices added'},...
   {'Solid line: true cutoffs for choosing A vs B, dotted line: implied cutoffs of human subjects'});
xlabel(sprintf('Prior probability \\pi_A'));
ylabel(sprintf('Number of balls marked N, n'));
hold off;

figure(8);
clf
hold on;
line([0 true_cutoffs(7)],[6 6],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(7)],[6 5],'Linewidth',1,'Color','k');
line([true_cutoffs(7) true_cutoffs(6)],[5 5],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(6)],[5 4],'Linewidth',1,'Color','k');
line([true_cutoffs(6) true_cutoffs(5)],[4 4],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(5)],[4 3],'Linewidth',1,'Color','k');
line([true_cutoffs(5) true_cutoffs(4)],[3 3],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(4)],[3 2],'Linewidth',1,'Color','k');
line([true_cutoffs(4) true_cutoffs(3)],[2 2],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(3)],[2 1],'Linewidth',1,'Color','k');
line([true_cutoffs(3) true_cutoffs(2)],[1 1],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(2)],[1 0],'Linewidth',1,'Color','k');
line([true_cutoffs(2) true_cutoffs(1)],[0 0],'Linewidth',1,'Color','k');
line([true_cutoffs(1) true_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k');
line([true_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k');

line([0 implied_human_cutoffs(7)],[6 6],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(7) implied_human_cutoffs(7)],[6 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(7) implied_human_cutoffs(6)],[5 5],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(6) implied_human_cutoffs(6)],[5 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(6) implied_human_cutoffs(5)],[4 4],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(5) implied_human_cutoffs(5)],[4 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(5) implied_human_cutoffs(4)],[3 3],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(4) implied_human_cutoffs(4)],[3 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(4) implied_human_cutoffs(3)],[2 2],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(3) implied_human_cutoffs(3)],[2 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(3) implied_human_cutoffs(2)],[1 1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(2) implied_human_cutoffs(2)],[1 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(2) implied_human_cutoffs(1)],[0 0],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(1) implied_human_cutoffs(1)],[0 -1],'Linewidth',1,'Color','k','LineStyle',':');
line([implied_human_cutoffs(1) 1],[-1 -1],'Linewidth',1,'Color','k','LineStyle',':');

text(.88,5.7,'Choose A');
text(.01,-.5,'Choose B');
text(.2,-.25,'Size of circles proportional to number of observations');
text(.2,-.5,'Shading of circles indicates fraction choosing A or B');
text(.2,-.75,'Light blue is 100% B, light red is 100% A, purple if 50% A, etc');

red=[1,0,0];
blue=[0,0,1];
ndraws=(0:nballs)';
npriors=[1/3; 1/2; 2/3];
total_nobs=0;
for nb=1:nballs+1
 for np=1:3
    frac=(npriors(np) > true_cutoffs(1+ndraws(nb)));
    nobs=sum(xdata(:,1)==ndraws(nb) & abs(xdata(:,2)-npriors(np))<.01);
    total_nobs=total_nobs+nobs;
    if (nobs)
    color=frac*red+(1-frac)*blue;
    %fprintf('balls %i prior %g frac %g nobs=%i\n',ndraws(nb),npriors(np),frac,nobs);
    ms=1+40*(nobs/600);
    plot(npriors(np),ndraws(nb),'o','Markersize',ms,'MarkerEdgeColor',color,'MarkerFaceColor',color);
    end
 end
end

title({'Decision regions implied by Bayes Rule, with choices of a perfect Bayesian added'},...
   {'Solid line: Bayesian cutoffs for choosing A vs B, dotted line: implied subject cutoffs'});
xlabel(sprintf('Prior probability \\pi_A'));
ylabel(sprintf('Number of balls marked N, n'));
hold off;

% compare performance of trained 2 level neural network with that from a support vector machine

posterior_a_trained=learning_how_to_learn.subjective_posterior_prob(training_xdata,trainedthetahat(2:4),model);
predicted_y_trained=(posterior_a_trained > 1/2);

for i=1:training_sample_size

   posterior_a_data(i,:)=binprobs_a*xdata(i,2);
   posterior_a_data(i,:)=posterior_a_data(i,:)./(posterior_a_data(i,:)+binprobs_b*(1-xdata(i,2)));
   predicted_y(i)=(posterior_a_data(i,xdata(i,1)+1) >= .5);
   predicted_y_trained(i)=(posterior_a_trained(i) >= .5);

end

svmmodel=fitcsvm(training_xdata,training_ydata,'holdout',.2);
[predicted_y_svm,score] = predict(svmmodel.Trained{1},training_xdata);

%[ydata predicted_y predicted_y_svm]
fprintf('Results for training sample of size %i\n',training_sample_size);
fprintf('prediction error rate from trained Bayes rule classifier: %g\n',sum(training_ydata ~= predicted_y_trained)/training_sample_size);
fprintf('prediction error rate from trained SVM classifier       : %g\n',sum(training_ydata ~= predicted_y_svm)/training_sample_size);

% plot last figure to show expected conditional loss from using Bayes Rule as the decision rule

figure(9);
clf;
x=(0:.01:1)';
sx=size(x,1);
y=zeros(sx,1);
y=x.*(x<=.5)+(1-x).*(x>.5);


hold on;
plot(x,y,'k-','Linewidth',2);
plot(posteriors(:,1),posteriors(:,5).*(1-posteriors(:,1))+(1-posteriors(:,5)).*posteriors(:,1),'r-','Linewidth',2);
plot(posteriors(:,1),posteriors(:,4).*(1-posteriors(:,1))+(1-posteriors(:,4)).*posteriors(:,1),'b-','Linewidth',2);
xlabel(sprintf('Posterior probability P(A|n,\\pi_A)'));
ylabel(sprintf('Expected loss (prediction error) given (n,\\pi_A)'));
legend('Perfect Bayesian','Human subjects','Trained machine learner','Location','Northwest');
title(sprintf('Expected Loss (Prediction Error) from Bayes Rule given (n,\\pi_A)'));
axis('tight');
hold off;

