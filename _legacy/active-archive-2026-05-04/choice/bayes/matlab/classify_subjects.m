% classify_subjects.m  program to compare the human subjects' choices of which bingo cage a sample of balls were drawn from
%                      to the classification produced by Bayes Rule for the experimental data from the paper
%                      "Are People Bayesian?" 1995 by El-Gamal and Grether
%                      John Rust, Georgetown University, February 2023

clear all;

load datastruct;  % this is a data structure that contains the data from the experiment
load structural_thetahat;

sample_type='all_pooled';  % choices: 'all_pooled' or 'all_individual'

model='prior_as_log_odds';  % 'llr_lpr' or 'prior_as_log_odds' or 'prior_linearly'
model='llr_lpr';
%model='prior_linearly';

if (strcmp(model,'llr_lpr'))  % in this specification xdata has first column equal to log(f(n|A)/f(n|B)) and 2nd column equal to log(pa/(1-pa)) where pa is the prior that cage A was used
   truetheta=[0; -1; -1];
else   % in this specification xdata has first column equal to n and 2nd column equal to prior
   truetheta=[6*(log(1/2)-log(1/3)); -log(2); 1];
end

options=optimoptions('fminunc','FunctionTolerance',1e-10,'Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');


redo=0;

   % load data and estimate the model
   [ydata,xdata]=learning_how_to_learn.prepare_data(datastruct,sample_type);
%load idx;
%ydata=ydata(find(idx==1));
%xdata=xdata(find(idx==1),:);
%structural_thetahat=[ 0.053576     -0.04452      -1.8593      -1.5606]';

   bayes_posterior_a=learning_how_to_learn.subjective_posterior_prob(xdata,truetheta,model); % this calculates the true posterior probabilities via Bayes Rule
   %llf_elgamal_grether_model=mean(log(1-averr/2)*(bayes_posterior_a<=1/2).*(ydata==0)+log(averr/2)*(bayes_posterior_a<=1/2).*(ydata==1));
   %llf_elgamal_grether_model=+llf_elgamal_grether_model+mean(log(1-averr/2)*(bayes_posterior_a>1/2).*(ydata==1)+log(averr/2)*(bayes_posterior_a>1/2).*(ydata==0));
   %err_rate=sum((bayes_posterior_a<=1/2).*(ydata==1)+(bayes_posterior_a>1/2).*(ydata==0));
   [llf_elgamal_grether_model,ic,total_obs,total_subjects,err_rate]=learning_how_to_learn.eg_lf_singletype([4 3 2],datastruct,'');
 
   fprintf('Estimating reduced-form binary logit model of subject choices using model specification %s\n',model);
   tic;
   [thetahat,llf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.blogit(ydata,xdata,theta,model),0*truetheta,options);
   toc

   fprintf('Estimating structural binary logit model of subject choices using model specification %s\n',model);
   tic;
   [structural_thetahat,sllf,exitflag,output,grad,hessian]=fminunc( @(theta) learning_how_to_learn.structural_blogit(ydata,xdata,theta),structural_thetahat,options);
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
   fprintf('Log-likelihood of structural logit model: %g\n',sllf);
   fprintf('\nMaximum likelihood estimation of a restricted "noisy Bayesian" model\n');
   [restricted_sigma,llfr]=fminbnd(@(x) learning_how_to_learn.structural_blogit(ydata,xdata,[x; truetheta]),0,1);
   llfr=-llfr;
   fprintf('Maximum likelihood estimate of sigma in restricted "noisy Bayesian model" model %g, log-likehood value: %g\n',restricted_sigma,llfr);
   fprintf('P-value of likeihood ratio test of noisy Bayesian model vs unrestricted structural logit model: %g\n',chi2cdf(2*(sllf-llfr),3,'upper'));
   
   toc

   llf_logit_model=-learning_how_to_learn.blogit(ydata,xdata,thetahat,model);
   lrstat=2*(llf_logit_model-llf_elgamal_grether_model);
   pval=cdf('chi2',lrstat,3,'upper');
   averr=err_rate; % estimated error rate from the single cutoff rule specification in El-Gamal and Grether 1995 (this differs from their paper
                   % due to having incomplete original data from their study (the PCCNO.pay file is messed up and was not included in datastruct)

   % now check the optimal 2 type model with 2 cutoff rules, Bayes Rule (4,3,2) and Representativeness rule (3,3,3) and calculate maximum likelihood error rate 
   [x,v]=fminbnd(@(x) -learning_how_to_learn.eg_lf_multitype(x,[4 3 2;3 3 3],datastruct,''),0,1);
   averr2=x;

fprintf('Results from the %s experiments: sample size %i (trials*subjects)  total number of subjects: %i for an average of %g trials per subject\n',...
    sample_type,total_obs,total_subjects,total_obs/total_subjects);
fprintf('Log-likelihood, El-Gamal and Grether model: %g error prob: %g  reduced-form logit log-likelihood: %g structural logit log-likelihood %g\n',...
    llf_elgamal_grether_model,err_rate,llf_logit_model,sllf);
fprintf('Log-likeihood ratio test:  El-Gamal and Grether model versus logit model (3 degrees of freedom): statistic (2*log-likelihood ratio) %g  p-value: %g\n',lrstat,pval);
fprintf('Information criterion value for single type El-Gamal and Grether Model: %g\n',ic);

   [llf_mt,ic_mt,total_obs,classified_subjects]=learning_how_to_learn.eg_lf_multitype(averr2,[4 3 2;3 3 3],datastruct,'');
   fprintf('Log-likelihood for 2 type El-Gamal and Grether model estimated by EC algorithm: %g  IC=%g estimated error probability: %g\n',llf_mt,ic_mt,averr2);
   fprintf('Subjects classified as Bayesian (4,3,2): %i Subjects classifed as using Representativeness rule (3,3,3): %i\n',sum(classified_subjects==1),sum(classified_subjects==2));
   fraction_bayesians=100*sum(classified_subjects==1)/numel(classified_subjects);
   fraction_representativeness=100*sum(classified_subjects==2)/numel(classified_subjects);
 

% calculate a matrix of posterior probabilities for cage A for all possible draw outcomes (0,...,6) and priors used in the
% experiment (1/3,1/2,2/3)

pa=2/3;  % probability of drawing a ball marked N from bingo cage A (4 N balls, 2 G balls)
pb=1/2;  % probability of drawing a ball marked N from bingo cage B (3 N balls, 3 G balls)
nballs=6;  % number of balls in each bingo cage (draws are done with replacement)

binprobs_a=binopdf((0:nballs)',nballs,pa);
binprobs_b=binopdf((0:nballs)',nballs,pb);

prior=[1/3; 1/2; 2/3];
nprior=size(prior,1);

posterior_a=zeros(nballs+1,nprior);  % posterior probability for outcome with 0 N balls in sample as a function of prior

for i=1:nprior;

   posterior_a(:,i)=binprobs_a*prior(i);
   posterior_a(:,i)=posterior_a(:,i)./(posterior_a(:,i)+binprobs_b*(1-prior(i)));

end

% now loop through data set and create a matrix with rows (n,prior,nsubjects,nchoices_bayes,bayes_choice,posterior_prob,pay)
% where n is the number of balls marked N, prior is the prior used, nsubjects is the number of subjects for that trial
% nchoices_bayes is the number of subjects making the same choice as Bayes Rule ,and bayes_choice is the choice
% of bingo cage from Bayes Rule (1 if the posterior indicates cage A is more likely, otherwise 0) and posterior_prob is
% the posterior probability of cage A under Bayes Rule for the given trial and the last element is an indicator for whether 


experiment_summary=[];
first7_summary=[];
last7_summary=[];

numexps=numel(datastruct);

for e=1:numexps;

   ntrials=size(datastruct(e).priors,1);
   nsubjects=size(datastruct(e).subjectchoices,1);
   pay=datastruct(e).pay;

   for trial=1:ntrials

      posterior_prob=posterior_a(datastruct(e).ndraws(trial)+1,find(prior==datastruct(e).priors(trial)/6));
      posterior_prob=floor(10000*posterior_prob)/10000;
      bayes_choice=(posterior_prob>=1/2);
      experiment_summary=[experiment_summary; ...
      [datastruct(e).ndraws(trial) datastruct(e).priors(trial) nsubjects sum(datastruct(e).subjectchoices(:,trial)) bayes_choice posterior_prob pay]]; 
 
      if (trial <= 7)
        first7_summary=[first7_summary; ...
        [datastruct(e).ndraws(trial) datastruct(e).priors(trial) nsubjects sum(datastruct(e).subjectchoices(:,trial)) bayes_choice posterior_prob pay]]; 
      end
      if (trial > ntrials-7)
        last7_summary=[last7_summary; ...
        [datastruct(e).ndraws(trial) datastruct(e).priors(trial) nsubjects sum(datastruct(e).subjectchoices(:,trial)) bayes_choice posterior_prob pay]]; 
      end
 
   end

end

% now find unique rows of the experiment summary and tabulate fraction of subject choices consistent with Bayes Rule for each distinct posterior probability 

experiment_posteriors=unique(experiment_summary(:,6));  % this provides the unique posterior probabilities in the experiment, sorted

representativeness_model=zeros(size(experiment_posteriors,1),1);  % this will contain the predicted choices of subjects following the representativness (3,3,3)
                                                                 % cutoff rule in El-Gamal and Grether 1995

nep=size(experiment_posteriors,1);
estimated_reduced_form_ccp=zeros(nep,1);
estimated_structural_ccp=zeros(nep,1);
estimated_noisy_bayesian_ccp=zeros(nep,1);
estimated_subjective_posterior=zeros(nep,1);

ec_predictions=zeros(nep,2);
ml_predictions=zeros(nep,2); 
fe_predictions=zeros(nep,2); 
load('ml_thetahat');
load('ec_thetahat');
load('fe_thetahat');
load('ml_tw');
load('ec_tw');
load('fe_tw');
ml_thetahat=ml_thetahat(2:7); % only use the last elements: first element codes the fraction of type 1 subjects

fraction_subjects_choosing_a=zeros(nep,1);
fraction_subjects_choosing_a_pay=zeros(nep,1);
fraction_subjects_choosing_a_nopay=zeros(nep,1);
n_subjects=zeros(nep,1);
n_subjects_pay=zeros(nep,1);
n_subjects_nopay=zeros(nep,1);

fraction_subjects_choosing_a_begin=zeros(nep,1);
n_subjects_begin=zeros(nep,1);
fraction_subjects_choosing_a_end=zeros(nep,1);
n_subjects_end=zeros(nep,1);

for i=1:nep

  all_data=experiment_summary(find(abs(experiment_posteriors(i)-experiment_summary(:,6)) <.01),:);
  fraction_subjects_choosing_a(i)=sum(all_data(:,4))/sum(all_data(:,3));
  n_subjects(i)=sum(all_data(:,3));

  below3=sum(all_data(find(all_data(:,1)<=3),3));
  above3=sum(all_data(find(all_data(:,1)>3),3));
  representativeness_model(i)=(below3*(averr2/2)+above3*(1-averr2/2))/n_subjects(i);

  axdata=[all_data(:,1) all_data(:,2)/nballs];
  orgdata=axdata;

  if (strcmp(model,'llr_lpr'))

    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    axdata(:,1)=llr(axdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    axdata(:,2)=log(axdata(:,2))-log(1-axdata(:,2));

  else
  
    axdata=xdata;
  
  end

  if (strcmp(model,'prior_as_log_odds'))

    axdata(:,2)=log(1-axdata(:,2))-log(axdata(:,2));

  end

  nad=size(all_data,1);
  for ni=1:nad;
   xthetahat=[1 axdata(ni,1) axdata(ni,2)]*thetahat;
   estimated_reduced_form_ccp(i)=estimated_reduced_form_ccp(i)+all_data(ni,3)/(1+exp(xthetahat));  

   estimated_structural_ccp(i)=estimated_structural_ccp(i)+all_data(ni,3)*learning_how_to_learn.structural_ccp(orgdata(ni,:),structural_thetahat);
   estimated_noisy_bayesian_ccp(i)=estimated_noisy_bayesian_ccp(i)+all_data(ni,3)*learning_how_to_learn.structural_ccp(orgdata(ni,:),[restricted_sigma; truetheta]);
   estimated_subjective_posterior(i)=estimated_subjective_posterior(i)+all_data(ni,3)*learning_how_to_learn.subjective_posterior_prob(orgdata(ni,:),structural_thetahat(2:4),model);

   ec_xthetahat1=[1 axdata(ni,1) axdata(ni,2)]*ec_thetahat(1:3);
   ec_xthetahat2=[1 axdata(ni,1) axdata(ni,2)]*ec_thetahat(4:6);
   ec_predictions(i,1)=ec_predictions(i,1)+all_data(ni,3)/(1+exp(ec_xthetahat1));  
   ec_predictions(i,2)=ec_predictions(i,2)+all_data(ni,3)/(1+exp(ec_xthetahat2));  

   ml_xthetahat1=[1 axdata(ni,1) axdata(ni,2)]*ml_thetahat(1:3);
   ml_xthetahat2=[1 axdata(ni,1) axdata(ni,2)]*ml_thetahat(4:6);
   ml_predictions(i,1)=ml_predictions(i,1)+all_data(ni,3)/(1+exp(ml_xthetahat1));  
   ml_predictions(i,2)=ml_predictions(i,2)+all_data(ni,3)/(1+exp(ml_xthetahat2));  

   fe_xthetahat1=[1 axdata(ni,1) axdata(ni,2)]*fe_thetahat(1:3);
   fe_xthetahat2=[1 axdata(ni,1) axdata(ni,2)]*fe_thetahat(4:6);
   %fe_xthetahat3=[1 axdata(ni,1) axdata(ni,2)]*fe_thetahat(7:9);
   fe_predictions(i,1)=fe_predictions(i,1)+all_data(ni,3)/(1+exp(fe_xthetahat1));  
   fe_predictions(i,2)=fe_predictions(i,2)+all_data(ni,3)/(1+exp(fe_xthetahat2));  
   %fe_predictions(i,3)=fe_predictions(i,3)+all_data(ni,3)/(1+exp(fe_xthetahat3));  

  end;

  estimated_reduced_form_ccp(i)=estimated_reduced_form_ccp(i)/sum(all_data(:,3));
  estimated_structural_ccp(i)=estimated_structural_ccp(i)/sum(all_data(:,3));
  estimated_noisy_bayesian_ccp(i)=estimated_noisy_bayesian_ccp(i)/sum(all_data(:,3));
  estimated_subjective_posterior(i)=estimated_subjective_posterior(i)/sum(all_data(:,3));
  ec_predictions(i,:)=ec_predictions(i,:)/sum(all_data(:,3));
  ml_predictions(i,:)=ml_predictions(i,:)/sum(all_data(:,3));
  fe_predictions(i,:)=fe_predictions(i,:)/sum(all_data(:,3));

  pay_data=experiment_summary(find(experiment_posteriors(i)==experiment_summary(:,6) & experiment_summary(:,7)==1),:);
  fraction_subjects_choosing_a_pay(i)=sum(pay_data(:,4))/sum(pay_data(:,3));
  n_subjects_pay(i)=sum(pay_data(:,3));

  nopay_data=experiment_summary(find(experiment_posteriors(i)==experiment_summary(:,6) & experiment_summary(:,7)==0),:);
  fraction_subjects_choosing_a_nopay(i)=sum(nopay_data(:,4))/sum(nopay_data(:,3));
  n_subjects_nopay(i)=sum(nopay_data(:,3));

  begin_data=first7_summary(find(abs(experiment_posteriors(i)-first7_summary(:,6))<.01),:);
  fraction_subjects_choosing_a_begin(i)=sum(begin_data(:,4))/sum(begin_data(:,3));
  n_subjects_begin(i)=sum(begin_data(:,3));

  end_data=last7_summary(find(abs(experiment_posteriors(i)-last7_summary(:,6))<.01),:);
  fraction_subjects_choosing_a_end(i)=sum(end_data(:,4))/sum(end_data(:,3));
  n_subjects_end(i)=sum(end_data(:,3));

end

fine_posterior=(0:.001:1)';
elgamal_grether_model=(averr/2)*(fine_posterior<=1/2)+(1-averr/2)*(fine_posterior>1/2);

elgamal_grether_model2=(averr2/2)*(fine_posterior<=1/2)+(1-averr2/2)*(fine_posterior>1/2);

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,fraction_subjects_choosing_a_pay,'r-','Linewidth',2);
plot(experiment_posteriors,fraction_subjects_choosing_a_nopay,'b-','Linewidth',2);
plot(fine_posterior,elgamal_grether_model,'g-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Fraction of subjects choosing cage A in El-Gamal and Grether 1995');
legend('All experiments','For pay experiments','No pay experiments','El-Gamal and Grether model','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,estimated_structural_ccp,'r-','Linewidth',2);
plot(experiment_posteriors,estimated_reduced_form_ccp,'b-','Linewidth',2);
plot(fine_posterior,elgamal_grether_model,'g-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Fraction of subjects choosing cage A in El-Gamal and Grether 1995');
legend('Fraction subjects choosing A','Structural logit model','Reduced-form logit model','El-Gamal and Grether model','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,estimated_structural_ccp,'r-','Linewidth',2);
plot(experiment_posteriors,estimated_noisy_bayesian_ccp,'b-','Linewidth',2);
plot(fine_posterior,elgamal_grether_model,'g-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Fraction of subjects choosing cage A in El-Gamal and Grether 1995');
legend('Fraction subjects choosing A','Structural logit model','Noisy Bayesian model','El-Gamal and Grether model','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
x=(0:.02:1)';
hold on;
plot(x,x,'k-','Linewidth',2);
plot(experiment_posteriors,estimated_subjective_posterior,'r-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Subjective and Bayes Rule Probabilities of choosing cage A');
title('Subjective Posterior beliefs vs Bayes Rule in El-Gamal and Grether Experiment');
legend('Bayes Rule Posterior','Subjective Posterior','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(fine_posterior,elgamal_grether_model2,'g-','Linewidth',2);
plot(experiment_posteriors,representativeness_model,'b-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Fraction of subjects choosing cage A in El-Gamal and Grether 1995');
legend('Fraction choosing A',sprintf('Bayes Rule (4,3,2) %3.1f%%',fraction_bayesians),sprintf('Rep Rule (3,3,3) %3.1f%%',fraction_representativeness)','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,fraction_subjects_choosing_a_begin,'r-','Linewidth',2);
plot(experiment_posteriors,fraction_subjects_choosing_a_end,'b-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Fraction of subjects choosing cage A in El-Gamal and Grether 1995');
legend('All experiments','First 7 trials','Last 7 trials','Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,fe_predictions(:,1),'b-','Linewidth',2);
plot(experiment_posteriors,fe_predictions(:,2),'r-','Linewidth',2);
%plot(experiment_posteriors,fe_predictions(:,3),'g-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Subject types identified by fixed effects/k-means clustering');
legend('Empirical frequency',sprintf('Type 1, %3.1f%%',100*fe_tw(1)),sprintf('Type 2, %3.1f%%',100*fe_tw(2)),'Location','Southeast');
%legend('Empirical frequency',sprintf('Type 1, %3.1f%%',100*fe_tw(1)),sprintf('Type 2, %3.1f%%',100*fe_tw(2)),sprintf('Type 3, %3.1f%%',100*fe_tw(3)),'Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,ml_predictions(:,1),'r-','Linewidth',2);
plot(experiment_posteriors,ml_predictions(:,2),'b-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Subject types identified by Heckman-Singer mixed logit');
legend('Empirical frequency',sprintf('Type 1, %3.1f%%',100*ml_tw(1)),sprintf('Type 2, %3.1f%%',100*ml_tw(2)),'Location','Southeast');
axis('square');
hold off;

myfigure;
clf;
hold on;
plot(experiment_posteriors,fraction_subjects_choosing_a,'k-','Linewidth',2);
plot(experiment_posteriors,ec_predictions(:,1),'r-','Linewidth',2);
plot(experiment_posteriors,ec_predictions(:,2),'b-','Linewidth',2);
line([0 .5],[0 0],'LineStyle',':','Linewidth',2);
line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
line([.5 1],[1 1],'LineStyle',':','Linewidth',2);
xlabel(sprintf('Posterior probability of cage A, P(A|n,\\pi_A)'));
ylabel('Fraction of subjects choosing cage A');
title('Subject types identified by the EC algorithm');
legend('Empirical frequency',sprintf('Type 1, %3.1f%%',100*ec_tw(1)),sprintf('Type 2, %3.1f%%',100*ec_tw(2)),'Location','Southeast');
axis('square');
hold off;

