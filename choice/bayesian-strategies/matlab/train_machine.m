% train_machine.m  program to "train the machine" (i.e. estimate binary logit classification model by maximum likelihood)
%                  John Rust, Georgetown University, February, 2022


dependent_variable='true_classification';  % select either `true_classification' or else 'bayes_rule_classification' or 'noisy_bayes_rule'
                                           % where the former (the default return from generate_taining_data) is the actual bingo cage that was selected to draw the sample
                                           % whereas bayes_rule_classification is the cage that a perfect bayesian decision maker would select after observing xdata
                                           % and noisy_bayes_rule the dependent variable is to select y=1 (cage A selected) with probability equal to the true posterior
                                           % probability that cage A was selected as the cage from which the sample was drawn
%dependent_variable='noisy_bayes_rule';

model='structural_logit';   % 'structural_logit' a structural binary logit model that depends on "subjective posterior beliefs" that the observed sample is drawn from urn A       
                            % 'llr_lpr'  is the logistic regression specification of the model using the log of the posterior odds ratio, which breaks
                            %            into two pieces, the log-likelihood ratio (llr) and the log-prior-odds-ratio (lpr). We also add a constant but notice
                            %            Bayes rule emerges as the coefficient vector (0,-1,-1) where both llr and lpr are given equal weight in the log-posterior odds ratio 
                            % 'prior_as_log_odds' nests true Bayes rule as a special case for the truetheta parameter vector below
                            % 'prior_linearly'  does not nest true Bayes Rule as a special case, and thus constitutes a misspecified model that can
                            %                   can only approximate the Bayesian posterior belief


errspec='logit';           % specification for the error term: 'probit' or 'logit'      

training_sample_size=10;

if (strcmp(model,'structural_logit'))
   truetheta=[0; 0; -1; -1];
elseif (strcmp(model,'llr_lpr'))  % in this specification xdata has first column equal to log(f(n|A)/f(n|B)) and 2nd column equal to log(pa/(1-pa)) where pa is the prior that cage A was used
   truetheta=[0; -1; -1];
else   % in this specification xdata has first column equal to n and 2nd column equal to prior
   truetheta=[6*(log(1/2)-log(1/3)); -log(2); 1];
end

options=optimoptions('fminunc','FunctionTolerance',1e-10,'Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective',...
    'MaxFunctionEvaluations',1000,'MaxIterations',1000);
%options=optimoptions('fminunc','FunctionTolerance',1e-10,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);

if (~exist('ydata'))
  fprintf('Generating new training sample with %i observations\n',training_sample_size);
  [ydata,xdata]=learning_how_to_learn.generate_training_data(training_sample_size,dependent_variable);
end


if (strcmp(model,'structural_logit'))
  tic;
  if (exist('thetahat'))
    starttheta=thetahat;
  else
    starttheta=truetheta;
    starttheta(1)=1;
  end
  [thetahat,llf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.structural_blogit(ydata,xdata,theta),starttheta,options);
  toc
elseif (strcmp(errspec,'logit'))
  tic;
  [thetahat,llf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.blogit(ydata,xdata,theta,model),0*truetheta,options);
  toc
else
  tic;
  [thetahat,llf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.bprobit(ydata,xdata,theta,model),0*truetheta,options);
  toc
end

if (exitflag)
  fprintf('training of learning machine has completed using %i training instances (i.e. observations)\n',training_sample_size);
  if (strcmp(model,'structural_logit'))
    llft=learning_how_to_learn.structural_blogit(ydata,xdata,truetheta);
    fprintf('converged log-likelihood value %g  log-likelihood at true parameters %g likelihood ratio test P-value: %g\n',-llf,-llft,chi2cdf(2*(llft-llf),numel(truetheta),'upper'));
  elseif (strcmp(errspec,'logit'))
    llft=learning_how_to_learn.blogit(ydata,xdata,truetheta,model);
    fprintf('converged log-likelihood value %g  log-likelihood at true parameters %g likelihood ratio test P-value: %g\n',-llf,-llft,chi2cdf(2*(llft-llf),numel(truetheta),'upper'));
  else
    llft=learning_how_to_learn.bprobit(ydata,xdata,truetheta,model);
    fprintf('converged log-likelihood value %g  log-likelihood at true parameters %g likelihood ratio test P-value: %g\n',-llf,-llft,chi2cdf(2*(llft-llf),numel(truetheta),'upper'));
 end
else
  fprintf('fminunc terminated with exitflag %i, algorithm may not have converged\n',exitflag);
end

stderr=sqrt(diag(inv(hessian)));
fprintf('\nEstimated coefficients by maximum likelihood using the likelihood function in learning_how_to_learn.logit and Matlab fminunc command\n');
fprintf('Estimated and true parameter vectors and std errors (last column)\n');
[thetahat truetheta stderr]

fprintf('gradient of log-likelihood with respect to parameters\n');
grad

% now compare to using Matlab's built-in logistic regression function, mnrfit, to estimate the model

if (~strcmp(model,'structural_logit'))

ydata1=1+ydata;
xdata1=xdata;
prior=xdata1(:,2);

if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata1(:,1)=llr(xdata1(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata1(:,2)=log(prior)-log(1-prior);

end

if (strcmp(model,'prior_as_log_odds'))
xdata1(:,2)=log(1-prior)-log(prior);  % prior log odds in last column of x matrix
end

fprintf('Estimated coefficients by logistic regression and Matlab mnrfit command\n');
fprintf('Estimated and true parameter vectors and std errors (last column)\n');
tic;
[b,dev,stats]=mnrfit(xdata1,ydata1);
toc
if (strcmp(errspec,'logit'))
  fprintf('mnrfit result:  llf: %g\n',learning_how_to_learn.blogit(ydata,xdata,b,model));
else
  fprintf('mnrfit result:  llf: %g\n',learning_how_to_learn.bprobit(ydata,xdata,b,model));
end
[b truetheta stats.se]

end

% plot results

true_posteriors=learning_how_to_learn.subjective_posterior_prob(xdata,truetheta(2:4),model);
predicted_choices=learning_how_to_learn.structural_ccp(xdata,thetahat);

sorted_data=sort([true_posteriors predicted_choices],1);
fine_posterior=(0:.001:1)';
sigma=thetahat(1);
nobs=numel(fine_posterior);
tpa=zeros(nobs,1);
nspa=(1-2*fine_posterior)/sigma;
ind=find(fine_posterior>1/2);
indc=find(fine_posterior<=1/2);
  if (numel(ind))
     pa(ind)=1./(1+exp(nspa(ind)));
  end
  if (numel(indc))
      pa(indc)=exp(-nspa(indc))./(1+exp(-nspa(indc)));
  end
pa=pa';


myfigure;
clf;
hold on;
plot(sorted_data(:,1),sorted_data(:,2),'s','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','k');
%plot(fine_posterior,pa,'-r','Linewidth',1);
%legend('Predicted choices at training data','Predictions for other posterior probs','Location','Northwest');
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel(sprintf('Predicted classifications after %i training observations',training_sample_size));
title('Choices of a two-level neural network after training');

