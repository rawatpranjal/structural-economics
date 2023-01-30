% learning_how_to_learn.m:  this program can be viewed as a machine learning
%                           algorithm that learns via "supervised learning" about how to classify
%                           samples drawn from one of two possible urns correctly by showing the
%                           samples that are drawn for different prior probabilities of drawing the
%                           urn A with 4 N balls and  2 G balls vs urn B with 3 N balls and 3 G balls
%                           with replacement as describe in the 1995 JASA paper by El-Gamal and Grether,
%                           "Are People Bayesian? Uncovering Behavioral Strategies"
%
%                           John Rust, Georgetown University, February 2023  

classdef learning_how_to_learn 

properties (Constant)
    debg = 'on'; % 'on', 'off', or 'detailed'   
  end%properties
methods (Static)

 function [ydata,xdata]=prepare_data(datastruct,group);

 % group is currently 'all_pooled' for all subjects in both the 
 %                    'all_pooled_california' for all subjects in the california experiments
 %                    'all_pooled_wisconsin' for all subjects in the wisconsin experiments
 %                    'all_indvidual' return [ydata,xddata] as cell arrays, where each array contains the [ydata,xdata] matrix for an individual subject 

 % modeltype is currently 'binary_logit_prior_in_log_odds_form' or
 %                        'binary_logit_prior_in_linear_form'
 %                        'binary_logit_n_and_prior_in_polynomial_form_k'  where k is an integer indexing the highest power polynomial allowed

  if (strcmp(group,'all_pooled'))

    ngroups=size(datastruct(1:7),2);   % fix me: this is a temporary restriction to the california (first 7 elements) until I extend the code to
                                       % use the datastruct.experiment_location field
    ydata=[];
    xdata=[];

    for g=1:ngroups
     
       priors=datastruct(g).priors';
       ndraws=datastruct(g).ndraws';
       subjectchoices=datastruct(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
       ydata=[ydata; subjectchoices(:)];
       ndraws=repmat(ndraws,nsubjects,1);
       priors=repmat(priors,nsubjects,1);
       xdata=[xdata; [ndraws(:) priors(:)/datastruct(g).nballs_prior_cage]];

    end

  elseif (strcmp(group,'all_pooled_california'))

    ngroups=numel(datastruct);
    ydata=[];
    xdata=[];

    for g=1:ngroups
     
       priors=datastruct(g).priors';
       nballs=datastruct(g).nballs_prior_cage;
       priors=priors/nballs;
       ndraws=datastruct(g).ndraws';
       subjectchoices=datastruct(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
       ydata=[ydata; subjectchoices(:)];
       ndraws=repmat(ndraws,nsubjects,1);
       priors=repmat(priors,nsubjects,1);
       xdata=[xdata; [ndraws(:) priors(:)]];

    end

  elseif (strcmp(group,'all_pay'))

    ngroups_total=numel(datastruct);
    ydata=[];
    xdata=[];
    ngroups=0;
    for g=1:ngroups_total
      if (~contains(datastruct(g).name,'NO.pay'))
        ngroups=ngroups+1;
        datastruct_pay(ngroups)=datastruct(g);
      end
    end

    for g=1:ngroups

       priors=datastruct_pay(g).priors';
       nballs=datastruct_pay(g).nballs_prior_cage;
       priors=priors/nballs;
       ndraws=datastruct_pay(g).ndraws';
       subjectchoices=datastruct_pay(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
       ydata=[ydata; subjectchoices(:)];
       ndraws=repmat(ndraws,nsubjects,1);
       priors=repmat(priors,nsubjects,1);
       xdata=[xdata; [ndraws(:) priors(:)]];

    end

  elseif (strcmp(group,'all_nopay'))

    ngroups_total=numel(datastruct);
    ydata=[];
    xdata=[];
    ngroups=0;
    for g=1:ngroups_total
      if (contains(datastruct(g).name,'NO.pay'))
        ngroups=ngroups+1;
        datastruct_nopay(ngroups)=datastruct(g);
      end
    end

    for g=1:ngroups

       priors=datastruct_nopay(g).priors';
       nballs=datastruct_nopay(g).nballs_prior_cage;
       priors=priors/nballs;
       ndraws=datastruct_nopay(g).ndraws';
       subjectchoices=datastruct_nopay(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
       ydata=[ydata; subjectchoices(:)];
       ndraws=repmat(ndraws,nsubjects,1);
       priors=repmat(priors,nsubjects,1);
       xdata=[xdata; [ndraws(:) priors(:)]];

    end

  elseif (strcmp(group,'UCLA') | strcmp(group,'PCC') | strcmp(group,'OXY') | strcmp(group,'CSULA'))

    ngroups_total=numel(datastruct);
    ydata=[];
    xdata=[];
    ngroups=0;
    for g=1:ngroups_total
      if (contains(datastruct(g).name,group))
        ngroups=ngroups+1;
        datastruct_school(ngroups)=datastruct(g);
      end
    end

    for g=1:ngroups

       priors=datastruct_school(g).priors';
       nballs=datastruct_school(g).nballs_prior_cage;
       priors=priors/nballs;
       ndraws=datastruct_school(g).ndraws';
       subjectchoices=datastruct_school(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
       ydata=[ydata; subjectchoices(:)];
       ndraws=repmat(ndraws,nsubjects,1);
       priors=repmat(priors,nsubjects,1);
       xdata=[xdata; [ndraws(:) priors(:)]];

    end

  else

   % in this case ydata and xdata are cell arrays of size 1 x nsubjects  and each contains the responses of a single subject

    ngroups=size(datastruct(1:7),2);   % fix me: this is a temporary restriction to the california (first 7 elements) until I extend the code to
                                       % use the datastruct.experiment_location field
    ydata={};   % now ydata and xdata are cell arrays, with each array element containing the data for an individual subject
    xdata={};   % hence this way of returning the data enables looping and estimating individual-specific models of subject choices

    cc=0;       % index for the cell arrays, one index per subject in the experiment

    do_all=0;
    if (strcmp(group,'all_individual'))
      do_all=1;
    end

    for g=1:ngroups

     if (do_all |  ...
        (endsWith(group,'UCLA') & contains(datastruct(g).name,'UCLA')) |  ...
        (endsWith(group,'OXY') & contains(datastruct(g).name,'OXY')) |  ...
        (endsWith(group,'PCC') & contains(datastruct(g).name,'PCC')) |  ...
        (endsWith(group,'CSULA') & contains(datastruct(g).name,'CSULA')))

       priors=datastruct(g).priors/datastruct(g).nballs_prior_cage;
       ndraws=datastruct(g).ndraws;
       subjectchoices=datastruct(g).subjectchoices;
       nsubjects=size(subjectchoices,1);
  
       for i=1:nsubjects
          cc=cc+1;
          ydata{cc}=subjectchoices(i,:)';
          xdata{cc}=[ndraws priors];
       end

     end

    end

  end % end of if-branch switch to prepare the data in the form and for the subsample requested

 end

 function [pa,dpa,hpa]=subjective_posterior_prob(xdata,theta,model) 

 % computes the estimated logit probability ("subjective posterior of bingo cage A") given "trained parameters" theta
 % If theta values are given by the "true values" this subjective probability coicides with the true posterior from Bayes Rule
 %
 % Outputs:
 %
 % pa    subjective posterior probability of choosing urn A
 % dpa   gradient of pa with respect to the theta parameters
 % hpa   hessian of pa with respect to the theta parameters
 %
 % Inputs: 
 %
 % xdata a k x 2 matrix, where k is the number of different values of x (n,prior) that the subjective prior is evaluated at
 %       The first column of xdata contains n, the number of balls drawn marked N, and the 2nd column is the prior probability of drawing from urn A
 % 
 % when k=1 (only evaluating at a single (n,prior) observation) then pa is a scalar, dpa is a 3x1 vector and hpa is a 3x3 matrix
 %          since there are actually 3 parameters in theta, where we allow for a constaint terms (theta(1)) plus coefficients for
 %          the columns of xdata (theta(2) and theta(3)).
 % when k>1 then pa is a kx1 vector, dpa is a 3xk matrix, and hpa is a 3x3xk three dimensional array

 nobs=size(xdata,1);

 prior=xdata(:,2);

 if (strcmp(model,'llr_lpr') | strcmp(model,'structural_logit'))   

                                 % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

 end

 if (strcmp(model,'prior_as_log_odds'))
 xdata(:,2)=log(1-prior)-log(prior);  % prior log odds in last column of x matrix
 end
 
 xdata=[ones(nobs,1) xdata];  % add a constant term

 xtheta=xdata*theta;
 pa=1./(1+exp(xtheta));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example

 if (nargout > 1)
    dpa=-((pa.*(1-pa))').*xdata';
 end

 if (nargout > 2)
   if (nobs==1)
    hpa=-dpa*(1-2*pa)*xdata;
   else
     hpa=zeros(3,3,nobs);
     for i=1:nobs
       hpa(:,:,i)=-dpa(:,i)*(1-2*pa(i))*xdata(i,:);
     end
   end
 end

 end

 function [bayes_inconsistent,posterior_inconsistent,non_monotonic]=...
 plot_individual_data(subject,subject_ydata,subject_xdata,subject_thetahat,subject_llf,doplot)

 % This function analyzes the responses of an individual subject, numbered subject from the California experiments. There are 221 possible
 % subjects in the datastruct we have. Running the prepare_data with the 'all_individual' data option returns the subject-specific ydata,xdata
 % matrices for individual subjects. Inputs are
 %
 % subject  an integer index of the subjects produced from the prepare_data command where each subject's (ydata,xdata) is held in a cell array
 %          indexed from 1 to the number of subjects in the given data (221 for all experiments done in California)
 % subject_ydata a vector with the 0/1 choices of the subject in the series of experiments they participated in, where 1 is choice of cage A
 % subject_xdata a k x 2 matrix where k is the number of experiments the subject participated and the first column is the integer number of
 %               balls drawn that are marked N from the chosen cage, and the 2nd column is the prior probability of choosing cage A to draw from.
 % subject_thetahat is the subject-specific 4 x 1 structural parameter vector from maximization of the structual_blogit likelihood function.
 % subject_llf is the optimized log-likelihood value corresponding to the estimated thetahat, subject_thetahat
 % doplot      if non-zero, do a plot of the subject choices for each of the three prior probabilities of drawing from cage A
 %
 % Outputs:
 %
 % bayes_inconsistent  the percentage of choices the subject made that are inconsistent with Bayes rule, (i.e. choose the cage for which
 %                     the true posterior probability of that cage exceeds 1/2)
 % posterior_inconsistent the fraction of choices the subject makes with internal inconsistencies (i.e. different choices for draws with
 %                     the same true posterior probability)
 % non_monotonic       the fraction of choices where the empirical frequency of choosing cage A are not monotonically increasing in the
 %                     true posterior probability of choosing cage A

 true_posteriors=learning_how_to_learn.subjective_posterior_prob(subject_xdata,[0 -1 -1]','llr_lpr');
 subjective_posteriors=learning_how_to_learn.subjective_posterior_prob(subject_xdata,subject_thetahat(2:4),'llr_lpr');
 choice_prob_a=learning_how_to_learn.structural_ccp(subject_xdata,subject_thetahat);
 bayes_choice=(true_posteriors > 1/2);

 xx=[subject_ydata subject_xdata true_posteriors subjective_posteriors choice_prob_a];

 ind1=find(abs(xx(:,3)-1/3)<.01); % indices of experiments where prior is 1/3
 ind2=find(abs(xx(:,3)-1/2)<.01); % indices of experiments where prior is 1/2
 ind3=find(abs(xx(:,3)-2/3)<.01); % indices of experiments where prior is 2/3

 k=numel(subject_ydata);
 total_bayes_inconsistent=0;
 total_posterior_inconsistent=0;
 total_non_monotonic=0;

 bayes_inconsistent=mean(bayes_choice(ind1) ~= subject_ydata(ind1));
 xx1=xx(ind1,:);
 xx1=sortrows(xx1,2);  % gather all trials where cage A was selected with prior prob 1/3
 posterior_inconsistent=0; 
 non_monotonic=0;
 xx1_c=[];
 ndraws=unique(xx1(:,2));
 for i=1:numel(ndraws)
   if (sum(xx1(:,2)==ndraws(i)) == 1)
     xx1_c=[xx1_c; xx1(find(xx1(:,2)==ndraws(i)),:)];
   else
     first_ind=min(find(xx1(:,2)==ndraws(i)));
     xx1_c=[xx1_c; xx1(first_ind,:)];
     xx1_c(end,1)=mean(xx1(find(xx1(:,2)==ndraws(i)),1));
     if (xx1_c(end,1) > 0 & xx1_c(end,1) < 1)
        posterior_inconsistent=posterior_inconsistent+1;
     end
   end
 end

 nc=size(xx1_c,1);
 for i=2:nc
   if (xx1_c(i-1,1) > xx1_c(i,1))
     non_monotonic=non_monotonic+1;
   end
 end

 total_non_monotonic=total_non_monotonic+non_monotonic;
 posterior_inconsistent=posterior_inconsistent/nc;
 non_monotonic=non_monotonic/nc;

 if (doplot)

 fignum=subject*10+1;
 figure(fignum);
 hold on;
 plot(xx1_c(:,4),xx1_c(:,1),'ro','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r');
 plot(xx1_c(:,4),xx1_c(:,6),'kx','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','k');
 plot(xx1_c(:,4),xx1_c(:,5),'r-','Linewidth',2);
 plot(xx1_c(:,4),xx1_c(:,4),'k-','Linewidth',2);
 axis('tight');
 xl=xlim;
 if (xl(2) > .5 & xl(1) < .5)
 line([xl(1) .5],[0 0],'LineStyle',':','Linewidth',2);
 line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
 line([.5 xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 elseif (xl(1) > .5)
 line([xl(1) xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 else
 line([xl(1) xl(2)],[0 0],'LineStyle',':','Linewidth',2);
 end
 yl=ylim;
 if (yl(1) == 0 & yl(2) == 1)
 text(xl(1)+4*(xl(2)-xl(1))/5,.23,sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.17,sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.12,sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.07,sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.02,sprintf('llf=%g',subject_llf));
 else
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.23*(yl(2)-yl(1)),sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.17*(yl(2)-yl(1)),sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.12*(yl(2)-yl(1)),sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.07*(yl(2)-yl(1)),sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.02*(yl(2)-yl(1)),sprintf('llf=%g',subject_llf));
 end
 legend('Fraction A chosen','Estimated CCP','Subjective posterior A','True posterior','Bayes choice prob A','Location','Northwest'); 
 xlabel(sprintf('Posterior probability P(A|n,\\pi_A)')); 
 ylabel(sprintf('Choice of A and subjective posterior probability P(A|n,\\pi_A)')); 
 title({sprintf('Subject %i choices for \\pi_A=1/3, %i observations, %i unique posteriors',subject,numel(ind1),nc),...
 sprintf('Percentage inconsistent/non-monotonic (%5.2f,%5.2f,%5.2f)',...
 100*bayes_inconsistent,100*posterior_inconsistent,100*non_monotonic)});
 axis('square');
 hold off;

 end

 bayes_inconsistent=mean(bayes_choice(ind2) ~= subject_ydata(ind2));
 xx2=xx(ind2,:);
 xx2=sortrows(xx2,2);  % gather all trials where cage A was selected with prior prob 1/2
 posterior_inconsistent=0; 
 non_monotonic=0;
 xx2_c=[];
 ndraws=unique(xx2(:,2));
 for i=1:numel(ndraws)
   if (sum(xx2(:,2)==ndraws(i)) == 1)
     xx2_c=[xx2_c; xx2(find(xx2(:,2)==ndraws(i)),:)];
   else
     first_ind=min(find(xx2(:,2)==ndraws(i)));
     xx2_c=[xx2_c; xx2(first_ind,:)];
     xx2_c(end,1)=mean(xx2(find(xx2(:,2)==ndraws(i)),1));
     if (xx2_c(end,1) > 0 & xx2_c(end,1) < 1)
        posterior_inconsistent=posterior_inconsistent+1;
     end
   end
 end

 nc=size(xx2_c,1);
 for i=2:nc
   if (xx2_c(i-1,1) > xx2_c(i,1))
     non_monotonic=non_monotonic+1;
   end
 end

 total_non_monotonic=total_non_monotonic+non_monotonic;
 total_posterior_inconsistent=total_posterior_inconsistent+posterior_inconsistent;
 posterior_inconsistent=posterior_inconsistent/nc;
 non_monotonic=non_monotonic/nc;

 if (doplot)

 fignum=subject*10+2;
 figure(fignum);
 hold on;
 plot(xx2_c(:,4),xx2_c(:,1),'ro','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r');
 plot(xx2_c(:,4),xx2_c(:,6),'kx','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','k');
 plot(xx2_c(:,4),xx2_c(:,5),'r-','Linewidth',2);
 plot(xx2_c(:,4),xx2_c(:,4),'k-','Linewidth',2);
 axis('tight');
 xl=xlim;
 if (xl(2) > .5 & xl(1) < .5)
 line([xl(1) .5],[0 0],'LineStyle',':','Linewidth',2);
 line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
 line([.5 xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 elseif (xl(1) > .5)
 line([xl(1) xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 else
 line([xl(1) xl(2)],[0 0],'LineStyle',':','Linewidth',2);
 end
 yl=ylim;
 if (yl(1) == 0 & yl(2) == 1)
 text(xl(1)+4*(xl(2)-xl(1))/5,.23,sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.17,sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.12,sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.07,sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.02,sprintf('llf=%g',subject_llf));
 else
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.23*(yl(2)-yl(1)),sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.17*(yl(2)-yl(1)),sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.12*(yl(2)-yl(1)),sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.07*(yl(2)-yl(1)),sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.02*(yl(2)-yl(1)),sprintf('llf=%g',subject_llf));
 end
 legend('Fraction A chosen','Estimated CCP','Subjective posterior A','True posterior','Bayes choice prob A','Location','Northwest'); 
 xlabel(sprintf('Posterior probability P(A|n,\\pi_A)')); 
 ylabel(sprintf('Choice of A and subjective posterior probability P(A|n,\\pi_A)')); 
 title({sprintf('Subject %i choices for \\pi_A=1/2, %i observations, %i unique posteriors',subject,numel(ind2),nc),...
 sprintf('Percentage inconsistent/non-monotonic (%5.2f,%5.2f,%5.2f)',...
 100*bayes_inconsistent,100*posterior_inconsistent,100*non_monotonic)});
 axis('square');
 hold off;

 end

 bayes_inconsistent=mean(bayes_choice(ind3) ~= subject_ydata(ind3));
 xx3=xx(ind3,:);
 xx3=sortrows(xx3,2);  % gather all trials where cage A was selected with prior prob 2/3
 posterior_inconsistent=0; 
 non_monotonic=0;
 xx3_c=[];
 ndraws=unique(xx3(:,2));
 for i=1:numel(ndraws)
   if (sum(xx3(:,2)==ndraws(i)) == 1)
     xx3_c=[xx3_c; xx3(find(xx3(:,2)==ndraws(i)),:)];
   else
     first_ind=min(find(xx3(:,2)==ndraws(i)));
     xx3_c=[xx3_c; xx3(first_ind,:)];
     xx3_c(end,1)=mean(xx3(find(xx3(:,2)==ndraws(i)),1));
     if (xx3_c(end,1) > 0 & xx3_c(end,1) < 1)
        posterior_inconsistent=posterior_inconsistent+1;
     end
   end
 end

 nc=size(xx3_c,1);
 for i=2:nc
   if (xx3_c(i-1,1) > xx3_c(i,1))
     non_monotonic=non_monotonic+1;
   end
 end

 total_non_monotonic=total_non_monotonic+non_monotonic;
 total_posterior_inconsistent=total_posterior_inconsistent+posterior_inconsistent;
 posterior_inconsistent=posterior_inconsistent/nc;
 non_monotonic=non_monotonic/nc;

 if (doplot)

 fignum=subject*10+3;
 figure(fignum);
 hold on;
 plot(xx3_c(:,4),xx3_c(:,1),'ro','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r');
 plot(xx3_c(:,4),xx3_c(:,6),'kx','Linewidth',2,'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','k');
 plot(xx3_c(:,4),xx3_c(:,5),'r-','Linewidth',2);
 plot(xx3_c(:,4),xx3_c(:,4),'k-','Linewidth',2);
 axis('tight');
 xl=xlim;
 if (xl(2) > .5 & xl(1) < .5)
 line([xl(1) .5],[0 0],'LineStyle',':','Linewidth',2);
 line([.5 .5],[0 1],'LineStyle',':','Linewidth',2);
 line([.5 xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 elseif (xl(1) > .5)
 line([xl(1) xl(2)],[1 1],'LineStyle',':','Linewidth',2);
 else
 line([xl(1) xl(2)],[0 0],'LineStyle',':','Linewidth',2);
 end
 yl=ylim;
 if (yl(1) == 0 & yl(2) == 1)
 text(xl(1)+4*(xl(2)-xl(1))/5,.23,sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.17,sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.12,sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.07,sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,.02,sprintf('llf=%g',subject_llf));
 else
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.23*(yl(2)-yl(1)),sprintf('\\sigma=%g',subject_thetahat(1)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.17*(yl(2)-yl(1)),sprintf('\\theta_1=%g',subject_thetahat(2)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.12*(yl(2)-yl(1)),sprintf('\\theta_2=%g',subject_thetahat(3)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.07*(yl(2)-yl(1)),sprintf('\\theta_3=%g',subject_thetahat(4)));
 text(xl(1)+4*(xl(2)-xl(1))/5,yl(1)+.02*(yl(2)-yl(1)),sprintf('llf=%g',subject_llf));
 end
 legend('Fraction A chosen','Estimated CCP','Subjective posterior A','True posterior','Bayes choice prob A','Location','Northwest'); 
 xlabel(sprintf('Posterior probability P(A|n,\\pi_A)')); 
 ylabel(sprintf('Choice of A and subjective posterior probability P(A|n,\\pi_A)')); 
 title({sprintf('Subject %i choices for \\pi_A=2/3, %i observations, %i unique posteriors',subject,numel(ind3),nc),...
 sprintf('Percentage inconsistent/non-monotonic (%5.2f,%5.2f,%5.2f)',...
 100*bayes_inconsistent,100*posterior_inconsistent,100*non_monotonic)});
 axis('square');
 hold off;

 end

 bayes_inconsistent=mean(bayes_choice ~= subject_ydata);
 non_monotonic=total_non_monotonic/k;
 posterior_inconsistent=total_posterior_inconsistent/k;

 end

 function [llf,dllf,hllf,im]=structural_blogit(ydata,xdata,theta)

 % This function can be regarded as computing the log-likelihood function for a "structural" binary logit model of subject choices.
 % This is a log-likelihood function for a structural binary logit model of subject choices with 4 parameters: the 3 parameters
 % entering the subjective_posterior_prob function plus a noise scale parameter sigma indexing the random "errors" in making a decision.
 % We assume that the subject choice is based on a "noisy subjective probability" of choosing urn A and so the subject chooses urn A
 % when the total utility of choosing it is higher, calculated as the sum of the subjective probability that urn A is the chosen urn
 % plus an extreme value shock indexed by the parameter sigma.  This model nested the perfect Bayes Rule decision rule when sigma=0
 % and the parmeters of the subjective choice model are (beta0,beta1,beta2)=(0,-1,-1). theta=(sigma,beta0,beta1,beta2).
 % This function computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for the 4 parameters of the binary logit model given binary 0/1 dependent variable ydata (1 if the subject's chosen urn is urn A) 
 % and x variables in xdata are the number of balls marked N that were drawn (first column) and the prior prob of drawing A (second column).

 % outputs:
 % llf is the log-likelihood function
 % dllf is the gradient of log-likelihood function with respect to the 4 model parameters theta
 % hllf is the 4x4 hessian matrix of the log-likelihood with respect to the 4 model parameters theta
 % im is the 4x4 information matrix of the log-likelihood: the outer-product of the gradients with respect to the 4 model parameters theta

 % inputs:
 % ydata is a vector of 0s and 1s that has nobs rows in it, where y=1 if subject chose urn A and y=0 if subject chose urn B
 % xdata is a matrix with nobs rows and 2 columns, where the first column contains n the number of balls drawn marked N and 2nd column is the prior prob of drawing urn A

 nobs=size(xdata,1);
 sigma=theta(1);
% if (sigma < 0)
%  fprintf('sigma=%g\n',sigma);
% end
 [spa,dspa,hspa]=learning_how_to_learn.subjective_posterior_prob(xdata,theta(2:4),'llr_lpr');

 % pa=1./(1+exp((1-2*spa)/sigma));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example
 %
 % Note: to improve numerical stability and accuracy, we do not actually evaluate pa via the formula above. Instead we split observations
 %       into those where the subjective probability is above 1/2 and those where it is below 1/2 and evaluate them separately to avoid
 %       loss of accuracy or program bombing due to possible overflow from evaluating (1-2*pa)/sigma if it takes on large positive values 
 %       Thus, we do not use the obvious way to write the log likelihood below
 %       llf=-sum(ydata.*logpa+(1-ydata).*log1mpa);
 %       and instead evaluate in a mathematically equivalent but numerically stable way below

  if (sigma == 0)

    pa=(spa>1/2); 
    if (sum(pa == ydata)==nobs)
      llf=0;
    else
      llf=-log(0);
    end
 
  else

    pa=zeros(nobs,1);
    logsum=zeros(nobs,1);
    nspa=(1-2*spa)/sigma;
    ind=find(spa>1/2);
    indc=find(spa<=1/2);
    if (numel(ind))
      pa(ind)=1./(1+exp(nspa(ind)));
      logsum(ind)=log(1+exp(nspa(ind)));
    end
    if (numel(indc))
      pa(indc)=exp(-nspa(indc))./(1+exp(-nspa(indc)));
      logsum(indc)=nspa(indc)+log(1+exp(-nspa(indc)));
    end
    llf=sum(logsum)-sum((1-ydata).*nspa);

  end
                                  
 if (nargout > 1)
   dllf=zeros(4,1);
   gradmat=zeros(nobs,4);
   gradmat(:,1)=(pa-ydata).*nspa;
   gradmat(:,2:4)=2*(pa-ydata).*dspa';
   dllf=sum(gradmat)'/sigma;
 end

 if (nargout > 2)
   hllf=zeros(4,4);
   hllf(1)=-2*dllf(1)/sigma+sum(pa.*(1-pa).*nspa.*nspa)/(sigma^2);
   fouroversigma=4/sigma;
   for i=1:nobs
     hllf(2:4,2:4)=hllf(2:4,2:4)+2*(pa(i)-ydata(i))*hspa(:,:,i)+fouroversigma*pa(i)*(1-pa(i))*dspa(:,i)*(dspa(:,i)');
   end
   hllf(1,2:4)=sigma*2*(sum(pa.*(1-pa).*nspa.*dspa')-sum((pa-ydata).*dspa'))/(sigma^2);
   hllf(1,2:4)=2*(sum(pa.*(1-pa).*nspa.*dspa')-sum((pa-ydata).*dspa'))/(sigma^2);
   hllf(2:4,1)=hllf(1,2:4)';
   hllf(2:4,2:4)=hllf(2:4,2:4)/sigma;
 end

 if (nargout > 3)
   im=gradmat'*gradmat/(sigma^2);
 end


 end

 function [pa]=structural_ccp(xdata,theta)

 % This function the "structural" binary logit conditional choice probability of choosing urn A given xdata (number n balls, prior) for 4x1 structural parameters theta
 % We assume that the subject choice is based on a "noisy subjective probability" of choosing urn A and so the subject chooses urn A
 % when the total utility of choosing it is higher, calculated as the sum of the subjective probability that urn A is the chosen urn
 % plus an extreme value shock indexed by the parameter sigma.  This model nested the perfect Bayes Rule decision rule when sigma=0
 % and the parmeters of the subjective choice model are (beta0,beta1,beta2)=(0,-1,-1). theta=(sigma,beta0,beta1,beta2).
 % This function computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for the 4 parameters of the binary logit model given binary 0/1 dependent variable ydata (1 if the subject's chosen urn is urn A) 
 % and x variables in xdata are the number of balls marked N that were drawn (first column) and the prior prob of drawing A (second column).

 % outputs:
 % structural_ccp is the nobsx1 vector of structural probabilities of choosing urn A implied by parameters theta for data in xdata

 % inputs:
 % xdata is a matrix with nobs rows and 2 columns, where the first column contains n the number of balls drawn marked N and 2nd column is the prior prob of drawing urn A
 % theta is a 4x1 vector of structural parameters where theta(1) is the sigma scaling or noise parameter

 nobs=size(xdata,1);

 sigma=theta(1);
 [spa,dspa]=learning_how_to_learn.subjective_posterior_prob(xdata,theta(2:4),'llr_lpr');

 % pa=1./(1+exp((1-2*spa)/sigma));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example
 %
 % Note: to improve numerical stability and accuracy, we do not actually evaluate pa via the formula above. Instead we split observations
 %       into those where the subjective probability is above 1/2 and those where it is below 1/2 and evaluate them separately to avoid
 %       loss of accuracy or program bombing due to possible overflow from evaluating (1-2*pa)/sigma if it takes on large positive values 
 %       Thus, we do not use the obvious way to write the log likelihood below
 %       llf=-sum(ydata.*logpa+(1-ydata).*log1mpa);
 %       and instead evaluate in a mathematically equivalent but numerically stable way below

 if (sigma == 0)
    pa=1*(spa>=1/2); 
    if (sum(pa == ydata)==nobs)
      llf=nobs;
    else
      llf=-log(0);
    end
 else 
    pa=zeros(nobs,1);
    logsum=zeros(nobs,1);
    nspa=(1-2*spa)/sigma;
    ind=find(spa>1/2);
    indc=find(spa<=1/2);
    if (numel(ind))
      pa(ind)=1./(1+exp(nspa(ind)));
      logsum(ind)=log(1+exp(nspa(ind)));
    end
    if (numel(indc))
      pa(indc)=exp(-nspa(indc))./(1+exp(-nspa(indc)));
      logsum(indc)=nspa(indc)+log(1+exp(-nspa(indc)));
    end
 end

 end

 function [cutoffs]=subjective_cutoffs(theta) 

 % computes the value of the prior probability of sampling from cage A that makes the "subjective posterior belief" equal to 1/2
 % for each of the seven possible draws of the number of balls marked N in the urn, (0,1,...,6) so cutoffs is a 7x1 vector with the
 % "cutoff" prior probabilities (so for any number of balls marked N, cutoffs(n+1) is the prior cutoff value, so any prior larger than 
 % this should lead the subjective decision maker to select cage A as the cage from which the sample of 6 balls was drawn from.
 %
 % Outputs:
 %
 % cutoffs: a 7x1 vector of the cutoff values of the prior probability of drawing from cage A that make the subjective posterior equal 1/2
 %
 % Inputs: 
 %
 % theta a 3 x 1 vector of the coefficients of the subjective posterior for (intercept, log(f(n|A)/f(n|B)), log(pi/(1-pi)))
 % 

 cutoffs=zeros(7,1);

 pa=2/3;
 pb=1/2;
 ndraws=6;
 outcomes=(0:ndraws)';
 llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));

 cutoffs=exp((-theta(1)-theta(2)*llr)/theta(3));
 cutoffs=cutoffs./(1+cutoffs);

 end


 function [llf,dllf,hllf,im]=blogit(ydata,xdata,theta,model)

 % This function can be regarded as computing the log-likelihood function for a "reduced form" logit model of subject choices.
 % See structural_blogit function for a structural binary logit log-likelihood function with 4 parameters: the 3 parameters
 % entering the subjective_posterior_prob function plus a noise scale parameter sigma indexing the random "errors" in making a decision.
 % This function computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for a 3 parameter binary logit model given binary 0/1 dependent variable ydata and x variables in xdata
 % model is a string indicating the specification: currently either 'prior_as_log_odds' or 'prior_linearly' or 'llr_plr'
 % In all cases we add a constant term, and the case prior_as_log_odds and prior_linearly the first column of xdata 
 % (before we add the constant) contains n the number of balls marked 'N; in the El-Gamal and Grether experiment.
 % When model is 'prior_as_log_odds' the second column contains log(1-prior)-log(prior) (the negative log-prior odds)
 % When model is 'prior_linearly' the second column is just the prior, entered linearly in the specification
 % When model is 'llr_lpr' the first column of xdata is assumed to contain log(f(n|A)/f(n|B)), the log-likelihood ratio
 % for drawing from cage A versus drawing from cage B, and the 2nd column of xdata contains log(prior/(1-prior)), the
 % log prior odds ratio. 

 % outputs:
 % llf is the log-likelihood function
 % dllf is the gradient of log-likelihood function with respect to the 3 model parameters theta
 % hllf is the 3x3 hessian matrix of the log-likelihood with respect to the 3 model parameters theta
 % im is the 3x3 information matrix of the log-likelihood: the outer-product of the gradients with respect to the 3 model parameters theta

 nobs=size(xdata,1);

 prior=xdata(:,2);
 if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

 end

 if (strcmp(model,'prior_as_log_odds'))
 xdata(:,2)=log(1-prior)-log(prior);  % negative of prior log odds in last column of x matrix
 end
 
 xdata=[ones(nobs,1) xdata];  % add a constant term

 xtheta=xdata*theta;
 pa=1./(1+exp(xtheta));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example

 logpa=log(pa);
 isinfpa=isinf(logpa);
 logpa(isinfpa)=-750;

 log1mpa=log(1-pa);
 isinf1mpa=isinf(log1mpa);
 log1mpa(isinf1mpa)=-750;
 
 %[ydata pa log(pa) log(1-pa) isinf(log(pa)) isinf(log(1-pa)) isinfpa logpa]

 %llf=-mean(ydata.*log(pa)+(1-ydata).*log(1-pa));
 llf=-sum(ydata.*logpa+(1-ydata).*log1mpa);

 if (nargout > 1)
 dllf=-sum(xdata.*(pa-ydata));
 end

 if (nargout > 2)
 hllf=((1-pa).*xdata)'*(pa.*xdata);
 end

 if (nargout > 3)
 im=(xdata.*(pa-ydata));
 im=im'*im;
 end

 end

 function [llf,dllf,hllf,im]=bprobit(ydata,xdata,theta,model)

 % computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for the binary probit model given binary 0/1 dependent variable ydata and x variables in xdata
 % model is a string indicating the specification: currently either 'prior_as_log_odds' or 'prior_linearly' or 'llr_plr'
 % In all cases we add a constant term, and the case prior_as_log_odds and prior_linearly the first column of xdata 
 % (before we add the constant) contains n the number of balls marked 'N; in the El-Gamal and Grether experiment.
 % When model is 'prior_as_log_odds' the second column contains log(1-prior)-log(prior) (the negative log-prior odds)
 % When model is 'prior_linearly' the second column is just the prior, entered linearly in the specification
 % When model is 'llr_lpr' the first column of xdata is assumed to contain log(f(n|A)/f(n|B)), the log-likelihood ratio
 % for drawing from cage A versus drawing from cage B, and the 2nd column of xdata contains log(prior/(1-prior)), the
 % log prior odds ratio

 % outputs:
 % llf is the log-likelihood function
 % dllf is the gradient of log-likelihood function with respect to the 3 model parameters theta
 % hllf is the 3x3 hessian matrix of the log-likelihood with respect to the 3 model parameters theta
 % im is the 3x3 information matrix of the log-likelihood: the outer-product of the gradients with respect to the 3 model parameters theta

 nobs=size(xdata,1);

 prior=xdata(:,2);
 if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

 end

 if (strcmp(model,'prior_as_log_odds'))
 xdata(:,2)=log(1-prior)-log(prior);  % negative of prior log odds in last column of x matrix
 end
 
 xdata=[ones(nobs,1) xdata];  % add a constant term

 xtheta=xdata*theta;
 pa=cdf('norm',xtheta,0,1);  % compute binary probit probability of selecting the first alternative, i.e. bingo cage A in this example

 llf=-sum(ydata.*log(pa)+(1-ydata).*log(1-pa));

 if (nargout > 1)
   pdftheta=pdf('norm',xtheta,0,1);
   dllf=-sum(xdata.*(ydata.*(pdftheta./pa)-(1-ydata).*(pdftheta./(1-pa))));
 end

 if (nargout > 2)
   hllf=zeros(3,3);
   for i=1:nobs
     ratio=pdftheta(i)/pa(i);
     ratio1=pdftheta(i)/(1-pa(i));
     hllf=hllf+(xdata(i,:)*xdata(i,:)')*((ratio^2+ratio*xtheta(i))*ydata(i)-(1-ydata(i))*(ratio1*xtheta(i)+ratio1^2));
   end
 end

 if (nargout > 3)
   im=(xdata.*(ydata.*(pdftheta./pa)-(1-ydata).*(pdftheta./(1-pa))));
   im=im'*im;
 end


 end

 function [llf,dllf,hllf]=mixed_blogit_WRONG(ydata,xdata,theta,ntypes,model)

 % NOTE: there is a problem in this implementation of mixed logit: can you spot the problem? 
 %
 % computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for the mixed binary logit model with ntypes types of individuals who have different choice/response probabilities 
 % given by type-specific binary logit models with 1/0 dependent variable ydata and x variables in xdata
 % model is a string indicating the specification: currently either 'prior_as_log_odds' or 'prior_linearly' or 'llr_plr'
 % In all cases we add a constant term, and the case prior_as_log_odds and prior_linearly the first column of xdata 
 % (before we add the constant) contains n the number of balls marked 'N; in the El-Gamal and Grether experiment.
 % When model is 'prior_as_log_odds' the second column contains log(1-prior)-log(prior) (the negative log-prior odds)
 % When model is 'prior_linearly' the second column is just the prior, entered linearly in the specification
 % When model is 'llr_lpr' the first column of xdata is assumed to contain log(f(n|A)/f(n|B)), the log-likelihood ratio
 % for drawing from cage A versus drawing from cage B, and the 2nd column of xdata contains log(prior/(1-prior)), the
 % log prior odds ratio
 %
 % Thus after adding a constant, xdata always has 3 columns and thus 3 coefficients per type.
 % With ntypes then theta has 3*ntypes+ntypes-1 elements, so the first ntypes-1 coefficients determine the probabilities
 % of the ntypes of individuals in the population, and the reamining 3*ntypes elements are the 3x1 coefficient vectors
 % for each of the ntypes types, stacked in order, from type 0, 1,...,ntypes-1. The first ntypes-1 coefficients are
 % logit coefficients for types 1,...,ntypes-1 where for type j, the probability is p_j=exp(theta_j)/(1+\sum_{j'=1}^{ntypes-1} exp(theta_j'})
 % and thus (theta_1,theta_2,...theta_{ntypes-1}) determine fhe fractions of each type in the population

 typeprobs=exp(theta(1:ntypes-1)); 
 typeprobs=[1 typeprobs']'./(1+sum(typeprobs));

 theta=reshape(theta(ntypes:end),3,ntypes);

 nobs=size(xdata,1);

 prior=xdata(:,2);
 if (strcmp(model,'prior_as_log_odds'))
 xdata(:,2)=log(1-prior)-log(prior);  % prior log odds in last column of x matrix
 end

 if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

 end
 
 xdata=[ones(nobs,1) xdata];  % add a constant term

 xtheta=xdata*theta;
 pa=1./(1+exp(xtheta));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example

 mpa=pa*typeprobs;

 logpa=log(mpa);
 isinfpa=isinf(logpa);
 logpa(isinfpa)=-750;
 log1mpa=log(1-mpa);
 isinf1mpa=isinf(log1mpa);
 log1mpa(isinf1mpa)=-750;
 
 %[ydata pa log(pa) log(1-pa) isinf(log(pa)) isinf(log(1-pa)) isinfpa logpa]

 %llf=-mean(ydata.*log(pa)+(1-ydata).*log(1-pa));
 llf=-mean(ydata.*logpa+(1-ydata).*log1mpa);

 if (nargout > 1)
 dllf=-mean(xdata.*(pa-ydata));
 end

 if (nargout > 2)
 hllf=((1-pa).*xdata)'*(pa.*xdata)/nobs;
 end

 end

 function [llf,dllf,hllf,typecount]=mixed_blogit(ycdata,xcdata,theta,ntypes,model)

 % Computes the log-likelihood (llf), gradients of the log-likeihood (dllf) and hessian of the log-likelihood (hllf)
 % for the mixed binary logit model with ntypes types of individuals who have different choice/response probabilities 
 % given by type-specific binary logit models with 1/0 dependent variable ydata and x variables in xdata
 % model is a string indicating the specification: currently either 'prior_as_log_odds' or 'prior_linearly' or 'llr_plr'
 % In all cases we add a constant term, and the case prior_as_log_odds and prior_linearly the first column of xdata 
 % (before we add the constant) contains n the number of balls marked 'N; in the El-Gamal and Grether experiment.
 % When model is 'prior_as_log_odds' the second column contains log(1-prior)-log(prior) (the negative log-prior odds)
 % When model is 'prior_linearly' the second column is just the prior, entered linearly in the specification
 % When model is 'llr_lpr' the first column of xdata is assumed to contain log(f(n|A)/f(n|B)), the log-likelihood ratio
 % for drawing from cage A versus drawing from cage B, and the 2nd column of xdata contains log(prior/(1-prior)), the
 % log prior odds ratio
 %
 % Thus after adding a constant, xdata always has 3 columns and thus 3 coefficients per type.
 % With ntypes then theta has 3*ntypes+ntypes-1 elements, so the first ntypes-1 coefficients determine the probabilities
 % of the ntypes of individuals in the population, and the reamining 3*ntypes elements are the 3x1 coefficient vectors
 % for each of the ntypes types, stacked in order, from type 0, 1,...,ntypes-1. The first ntypes-1 coefficients are
 % logit coefficients for types 1,...,ntypes-1 where for type j, the probability is p_j=exp(theta_j)/(1+\sum_{j'=1}^{ntypes-1} exp(theta_j'})
 % and thus (theta_1,theta_2,...theta_{ntypes-1}) determine fhe fractions of each type in the population

 dim_theta=size(theta,1);

 typeprobs=exp(theta(1:ntypes-1)); 
 typeprobs=[1 typeprobs']'./(1+sum(typeprobs));

 theta=reshape(theta(ntypes:end),3,ntypes);

 nsubjects=size(ycdata,2);

 llf=0;
 dllf=zeros(dim_theta,1);
 hllf=zeros(dim_theta,dim_theta);

 for subject=1:nsubjects

   ydata=ycdata{subject};
   xdata=xcdata{subject};
   nobs=size(ydata,1);

   prior=xdata(:,2);

   if (strcmp(model,'prior_as_log_odds'))
     xdata(:,2)=log(1-prior)-log(prior);  % prior log odds in last column of x matrix
   end

   if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

   end
 
   xdata=[ones(nobs,1) xdata];  % add a constant term

   xtheta=xdata*theta;
   pa=1./(1+exp(xtheta));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example

   mpa=pa*typeprobs;
   logpa=log(mpa);
   log1mpa=log(1-mpa);
 
   subject_llf=-sum(ydata.*logpa+(1-ydata).*log1mpa);
   llf=llf+subject_llf;

   if (nargout > 1)
     dllf=zeros(3*ntypes,1);
   end

   if (nargout > 2)
     hllf=zeros(3*ntypes,3*ntypes);
   end

 end

 end

 function [llf,dllf,hllf,typecount]=ec_blogit(ycdata,xcdata,theta,ntypes,model)

 % EC (estimation-classification) likelihood function. Similar to mixed_blogit function except for each person, the
 % function evaluates the person-specific likelihood contribution over the different types of individuals and picks
 % as the likelihood contribution the type with the highest person-specific likelihood.  Also returns the number of
 % observations in each category, so as a fraction of all subjects, these are analogous to the mixture weights in mixed_blogit.
 % In this function ydata and xdata are cell arrays, with each element containing the person-specific ydata and xdata matrices.
 % In all cases we add a constant term, and the case prior_as_log_odds and prior_linearly the first column of xdata 
 % (before we add the constant) contains n the number of balls marked 'N; in the El-Gamal and Grether experiment.
 % When model is 'prior_as_log_odds' the second column contains log(1-prior)-log(prior) (the negative log-prior odds)
 % When model is 'prior_linearly' the second column is just the prior, entered linearly in the specification
 % When model is 'llr_lpr' the first column of xdata is assumed to contain log(f(n|A)/f(n|B)), the log-likelihood ratio
 % for drawing from cage A versus drawing from cage B, and the 2nd column of xdata contains log(prior/(1-prior)), the
 % log prior odds ratio
 %
 % Thus after adding a constant, xdata always has 3 columns and thus 3 coefficients per type.
 % With ntypes then theta has 3*ntypes elements.

 theta=reshape(theta,3,ntypes);

 nsubjects=size(ycdata,2);

 llf=0;
 dllf=zeros(3*ntypes,1);
 hllf=zeros(3*ntypes,3*ntypes);
 typecount=zeros(ntypes,1);

 for subject=1:nsubjects

   ydata=ycdata{subject};
   xdata=xcdata{subject};
   nobs=size(ydata,1);
   prior=xdata(:,2);
   if (strcmp(model,'prior_as_log_odds'))
     xdata(:,2)=log(1-prior)-log(prior);  % prior log odds in last column of x matrix
   end

   if (strcmp(model,'llr_lpr'))    % fix me: need to differentiate between wisconsin experiments that had cages with 10 balls in them
                                 % the code below only applies to the California experiments where each cage had 6 balls and also two
                                 % of the Wisconsin experiments where the same cage design as California was maintained
    pa=2/3;
    pb=1/2;
    ndraws=6;
    outcomes=(0:ndraws)';
    llr=log(binopdf(outcomes,ndraws,pa))-log(binopdf(outcomes,ndraws,pb));
    xdata(:,1)=llr(xdata(:,1)+1);  % recode the first column of xdata as the log-likeihood ratio log(f(n|A)/f(n|B)) not n
    xdata(:,2)=log(prior)-log(1-prior);

   end
 
   xdata=[ones(nobs,1) xdata];  % add a constant term

   xtheta=xdata*theta;
   pa=1./(1+exp(xtheta));  % compute binary logit probability of selecting the first alternative, i.e. bingo cage A in this example

   logpa=log(pa);
   log1mpa=log(1-pa);
 
   llft=-sum(ydata.*logpa+(1-ydata).*log1mpa);
   mllft=min(llft);
   llf=llf+mllft;

   ind=find(mllft==llft);
   typecount(ind)=typecount(ind)+1;

   if (nargout > 1)
     subject_dllf=zeros(3*ntypes,1);
     subject_dllf(3*(ind-1)+1:3*ind)=-sum(xdata.*(pa(ind)-ydata));
     dllf=dllf+subject_dllf;
   end

   if (nargout > 2)
     hllf=zeros(3*ntypes,3*ntypes);
   end

 end

 end

 function [ydata,xdata]=generate_training_data(ntrain,dependent_variable)

 % inputs: ntrain      integer size of the "training sample" to generate to train the machine learning algorithm.  This will result in random samples
 %                     being drawn along with an indicator d which equals 1 if the sample was drawn from urn A otherwise 0 
 % dependent_variable  a string that equals either `true_classification' or else 'bayes_rule_classification'
 %                     where the former (the default return from generate_taining_data) is the actual bingo cage that was selected
 %                     whereas bayes_rule_classification is the cage that a perfect bayesian decision maker would select after observing xdata

 % outputs: 
 %         ydata a vector (ntrain x 1) equal to 1 if bingo cage A was used to draw the sample of balls with replacement, or 0 if drawn from B
 %                                     if dependent_variable is set to `true_classification' otherwise 
 %                                     it equals 0 if the posterior probability of A is less than 1/2 and 1 otherwise when dependent_variable is bayes_rule_classification
 %         xdata a matrix (ntrain x 2) whose first column contains n, the number of balls marked with an N from the selected bingo cage   
 %                                     and the 2nd column is the prior probability of drawing from cage A

 pa=2/3;  % binomial probability for urn (or bingo cage) A:  4N balls over 6 balls total, or 2/3 probability of drawing an N
 pb=1/2;  % binomial probability for urn (or bingo cage) B:  3N balls over 6 balls total, or 1/2 probability of drawing an N

 ndraws=6;  % number of draws from each urn (with replacement)

 xdata=zeros(ntrain,2);  % the data consist of the number of balls marked N that were drawn and the prior probability
                         % used to draw the sample
 ydata=zeros(ntrain,1);  % the ydata is the "dependent variable" d, showing which urn the sample was drawn from

 xdata(:,2)=rand(ntrain,1);  % here we randomly generate uniform(0,1) random numbers to be the prior probabilities used
                             % to train the machine 
   for i=1:ntrain

     % first we select the bingo cage, according to the true model using the prior probability stored in xdata(i,2)
 
     ydata(i)=(rand(1,1)<=xdata(i,2));  % if this is 1 then use urn A to draw the number of balls marked N from, otherwise
                                        % draw from urn B 

     % then given the chosen bingo cage, we generate a draw from it representing the result of drawing nballs with replacement

     if (ydata(i))
       xdata(i,1)=binornd(ndraws,pa,1);
     else
       xdata(i,1)=binornd(ndraws,pb,1);
     end

     % finally we redefine the dependent variable if the dependent_variable is set to bayes_rule_classification

        % calculate the posterior probability given the prior in xdata(i,2) and number of N balls drawn in xdata(i,1)

        ppa=binopdf(xdata(i,1),ndraws,pa)*xdata(i,2);
        den=ppa+binopdf(xdata(i,1),ndraws,pb)*(1-xdata(i,2));
        ppa=ppa/den;

        ydata(i)=(ppa >= 1/2);

        if (strcmp(dependent_variable,'noisy_bayes_rule'))

           ydata(i)=(rand(1,1) <= ppa);

        end 

   end  % end of do-loop to generate training data

 end % end of function generate_training_data 

 function [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_singletype(cr,datastruct,school)

 % eg_lf_singletype.m: subject likelihood in the El-Gamal and Grether model with a single "type" (decision rule) and
 %                     common estimated error rate
 %                     John Rust, Georgetown University, February, 2022
 %
 % Inputs:
 %
 % cr  a 3x1 vector of integer cutoffs: when there is no error, subject chooses urn A if number of N's under prior i (i=1,2,3)
 %     exceeds cr(i)
 % datastruct an array of structures holding the results for different experiments
 % school  a string variable that is empty, '', to evaluate for all schools, or a school abbreviation ('CSULA','OXY','UCLA','PCC')
 %         if you want to restrict the evaluation of the likelihood to only subjects of specific schools (see Table 2 of El-Gamal and Grether, 1995)


    numexps=numel(datastruct);
    llf=0;
    llf_homo=0;
    total_consistent=0;
    total_obs=0;
    total_subjects=0;

    k=size(cr,1);

    for e=1:numexps;    % loop over all experiments to calculate likelihood
      if (startsWith(datastruct(e).name,school) | strcmp(school,''))    % if the school string is provided, then likelihood is calculated for given school
        nsubjects=size(datastruct(e).subjectchoices,1);
        priors=datastruct(e).priors;
        ntrials=numel(priors);
        ndraws=datastruct(e).ndraws;

        cr_prior=cr(priors-1)';  % trial specific cutoff depending on the prior in that trails
        cr_choice=(ndraws > cr_prior)';  % predicted choices for each trial implied by the cutoff rule
        consistent_obs=sum((cr_choice == datastruct(e).subjectchoices)');
        err_rates=min(1,2*(1-sum(consistent_obs)/(nsubjects*ntrials)));

        total_subjects=total_subjects+nsubjects;
        total_consistent=total_consistent+sum(consistent_obs);
        total_obs=total_obs+nsubjects*ntrials;

        llf=llf+sum(consistent_obs.*log(1-err_rates/2)+(ntrials-consistent_obs).*log(err_rates/2));
      end
    end

    err_rate=2*(1-total_consistent/total_obs);

    llf=total_consistent*log(1-err_rate/2)+(total_obs-total_consistent)*log(err_rate/2);

    ic=llf-3*k*log(8)-k*log(2)-total_subjects*log(k);

 end % end of function eg_lf_singletype
      
 function [llf_cr,total_obs,total_subjects,err_rates,cutoffs,maxinfo,maxcutoffs]=eg_fixed_effects(datastruct,school)

 % eg_fixed_effects.m: fixed effects estimation of the El-Gamal and Grether model: each subject is assigned the maximum 
 %                     likelihood cutoff rule and error rate that maximizes the subject-specific likelihood
 %                     Note: to speed up the estimation, we only search over a subset of the 512 possible cutoff rules where
 %                     each cutoff (c1,c2,c3) is restricted to be an integer from 1 to 5
 %                     John Rust, Georgetown University, February, 2022
 %
 % Inputs:
 %
 % datastruct an array of structures holding the results for different experiments
 % school  a string variable that is empty, '', to evaluate for all schools, or a school abbreviation ('CSULA','OXY','UCLA','PCC')
 %         if you want to restrict the evaluation of the likelihood to only subjects of specific schools (see Table 2 of El-Gamal and Grether, 1995)

    numexps=numel(datastruct);
    llf_cr=[];;
    total_obs=0;
    total_subjects=0;
    err_rates=[];
    cutoffs=[];
    crlist=[];
    maxcutoffs=[];
    maxinfo=struct;

    for c1=-1:6
     for c2=-1:6
      for c3=-1:6
        crlist=[crlist; [c1 c2 c3]];
      end
     end
    end

    ncr=size(crlist,1);
    sc=0;
    bayes_consistent=0;
    perfect_bayes_consistent=0;

    for e=1:numexps;    % loop over all experiments to calculate likelihood
      if (startsWith(datastruct(e).name,school) | strcmp(school,''))    % if the school string is provided, then likelihood is calculated for given school

        nsubjects=size(datastruct(e).subjectchoices,1);
        priors=datastruct(e).priors;
        ntrials=numel(priors);
        ndraws=datastruct(e).ndraws;
        total_subjects=total_subjects+nsubjects;
        total_obs=total_obs+nsubjects*ntrials;

        for i=1:nsubjects
         sc=sc+1;
         llv=zeros(ncr,1);

         for r=1:ncr

           cr=crlist(r,:);
           cr_prior=cr(priors-1)';  % trial specific cutoff depending on the prior in that trails
           cr_choice=(ndraws > cr_prior)';  % predicted choices for each trial implied by the cutoff rule
           consistent_obs=sum((cr_choice == datastruct(e).subjectchoices(i,:)));
           if (consistent_obs == ntrials)
             err_rate=0;
             llf=0;
           else
             err_rate=min(1,2*(1-sum(consistent_obs)/ntrials));
             llf=sum(consistent_obs*log(1-err_rate/2)+(ntrials-consistent_obs)*log(err_rate/2));
           end
           llv(r)=llf;

           if (r == 1)
              mll=llf;
              bestcr=cr;
              best_err_rate=err_rate;
           else
              if (llf > mll)
                mll=llf;
                bestcr=cr;
                best_err_rate=err_rate;
              end
           end

         end
    
         llf_cr=[llf_cr; mll];
         err_rates=[err_rates; best_err_rate];
         cutoffs=[cutoffs; bestcr];
         nmax=sum(llv==mll);
         maxinfo(sc).llf=mll;
         maxinfo(sc).nmax=nmax;
         maxinfo(sc).optcutoffs=crlist(find(llv==mll),:);
         maxcutoffs=[maxcutoffs; maxinfo(sc).optcutoffs];

         % use the Bayes Rule cutoffs if they are among the set of optimizing cutoff rules

         if (sum((sum((maxinfo(sc).optcutoffs == [4 3 2])')') == 3) > 0)
            cutoffs(end,:)=[4 3 2];
            bestcr=[4 3 2];
            bayes_consistent=bayes_consistent+1;
            if (best_err_rate == 0)
              perfect_bayes_consistent=perfect_bayes_consistent+1;
            end
         end

         % fprintf('Experiment %i ntrials %i subject %i sc %i  llf=%g  err_rate=%g cr=(%i,%i,%i)\n',e,ntrials,i,sc,mll,best_err_rate,bestcr);

        end % end of loop over subjects in a given school/pay/experiment (element of datastruct array)
      end % end of if statement over which elements of datastruct to include
    end % end of loop over all elements in datastruct
    fprintf('%i out of %i subjects were best classified as Bayesians with cutoff rule [4 3 2] maximizing the likelihood\n',bayes_consistent,sc);
    fprintf('%i subjects are perfect Bayesians: i.e. all of their choices are consistent with Bayes rule\n',perfect_bayes_consistent);

 end % end of function eg_fixed_effects
      
 function [llf,ic,total_obs,classified_subjects]=eg_lf_multitype(err_rate,cr,datastruct,school)

 % eg_lf_multitype.m: subject likelihood in the El-Gamal and Grether model with multiple "types" (cutoff rules) and
 %                    common estimated error rate
 %                     John Rust, Georgetown University, February, 2022
 %
 % Inputs:
 %
 % err_rate a scalar common probability of "deviating" and randomly choosing an answer that we assume all subjects are doing
 % cr  a kx3 matrix of integer cutoffs: when there is no error, subject chooses urn A if number of N's under prior i (i=1,2,3)
 %     exceeds cr(k,i) under cutoff rule k.
 % datastruct an array of structures holding the results for different experiments
 % school  a string variable that is empty, '', to evaluate for all schools, or a school abbreviation ('CSULA','OXY','UCLA','PCC')
 %         if you want to restrict the evaluation of the likelihood to only subjects of specific schools (see Table 2 of El-Gamal and Grether, 1995)
 
 % Outputs:
 % llf  the log likelihood from the EC algorithm evaluated at the error rate err_rate
 % ic   the information criteriof (see El-Gamal and Grether 1995 for its definition, but it is a penalized likelihood, penalized for number of parameters
 % total_obs total number of subject/trial observations
 % classified_subjects  a vector that is of length total_subjects x 1 that indexes the cutoff rule that best predicts the subject's choices
 
    numexps=numel(datastruct);
    total_consistent=0;
    total_obs=0;
    llf=0;
    classified_subjects=[];

    k=size(cr,1);

    for e=1:numexps;    % loop over all experiments to calculate likelihood
      if (startsWith(datastruct(e).name,school))    % if the school string is provided, then likelihood is calculated for given school

        nsubjects=size(datastruct(e).subjectchoices,1);
        priors=datastruct(e).priors;
        ntrials=numel(priors);
        ndraws=datastruct(e).ndraws;
        total_obs=total_obs+nsubjects*ntrials;
        llfm=zeros(k,nsubjects);

        for t=1:k  % this do loop is the core of the EC algorithm: it computes subject-specific likelihoods for each cutoff rule in cr
          cr_prior=cr(t,priors-1)';  % trial specific cutoff depending on the prior in each trials, for cutoff rule t
          cr_choice=(ndraws > cr_prior)';  % predicted choices for each trial implied by cutoff rule t
          consistent_obs=sum((cr_choice == datastruct(e).subjectchoices)');
          llfm(t,:)=(consistent_obs*log(1-err_rate/2)+(ntrials-consistent_obs)*log(err_rate/2))';
        end
        if (k > 1)
          mllf=max(llfm);  % now we check which cutoff rule has the highest subject-specific likelihood and assign that rule to the subjects
          [ii,j]=find(llfm==mllf);
          [c,ia,ic]=unique(j,'stable');  % this code is necessary in case of ties, then the find command will return all indices where llfm is maximized
                                         % so the unique command picks out the first index in the matrix llfm where the maximum is attained, i.e. when
                                         % there are cutoff rules that lead to the same likelihood, we classify the subjects with the cutoff rule with the
                                         % lowest index in the matrix of cutoffs, cr
          cs=ii(ia);                     % this is the index of the cutoff rule with the highest likelihood for each subject given the current err_rate
        else
          cs=ones(nsubjects,1);
          mllf=llfm;
        end
 %  The code below is a slower way to do the above, in a do loop over each column of the llfm matrix
 %        cs=zeros(nsubjects,1);
 %        for i=1:nsubjects
 %           cs(i)=find(llfm(:,i)==mllf(i),1); 
 %        end
 %[cs ii(ia)]
 %sum(cs ~= ii(ia))
        llf=llf+sum(mllf);
        classified_subjects=[classified_subjects; cs];
      end
    end
   
    total_subjects=numel(classified_subjects);
    ic=llf-3*k*log(8)-k*log(2)-total_subjects*log(k);
      
 end % end of function eg_lf_multitype

 function [mll,mic,err_rate_best]=estimate_onetype_eg_model(datastruct,subsample);

 % estimate_onetype_eg_model.m  estimates the single type subcase of the El Gamal and Grether model,
 %                              compare to table 2 top rows in their 1995 paper, "Are People Bayesian?" in JASA
 
 % note that the function [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_singletype([c1 c2 c3],datastruct,subsample) 
 % calculates the log likelihood (llf), information criterion and the maximum likelihood pooled error rate for cutoff rule [c1 c2 c3]
 % where these are integers between -1 and 6 (see page 1139 of El Gamal and Grether for further explanation). The likelihood
 % routine returns the maximum likelihood estimate of the common error rate, err_rat, that all subbjects are presumed to behave
 % according to, in conjunction with the cutoff rule [c1 c2 c3] where the 3 values index the 3 possible priors (1/3,1/2,2/3) used
 % in the experiment. This program does a more localized discrete search for cutoffs from 1 to 5 and does it as a brute force
 % search in a simple do-loop.

   llv=zeros(125,1);
   icv=zeros(125,1);
   errv=zeros(125,1);
   i=0;

   for c1=1:5
     for c2=1:5
       for c3=1:5
           i=i+1;
           [llf,ic,total_obs,total_subjects,err_rate]=learning_how_to_learn.eg_lf_singletype([c1 c2 c3],datastruct,subsample);
           llv(i)=llf;
           icv(i)=ic;
           errv(i)=err_rate;
           if (i == 1)
              mll=llf;
              mic=ic;
              err_rate_best=err_rate;
              bestc=[c1 c2 c3];
              bestic=[c1 c2 c3];
           else
              if (llf > mll)
                mll=llf;
                err_rate_best=err_rate;
                bestc=[c1 c2 c3];
              end
              if (ic > mic)
                mic=ic;
                bestic=[c1 c2 c3];
              end
           end
           fprintf('%i searching (c1,c2,c3)=(%i,%i,%i) llf=%g ic=%g  best likelihood so far: %g best ic %g  error_rate=%g\n',i,c1,c2,c3,llf,ic,mll,mic,err_rate_best); 
       end
     end
   end
   if (strcmp(subsample,''))
     fprintf('\nSummary of maximum likelihood estimation of homogeneous cutoff rule using full El-Gamal and Grether California subject pool\n');
   else
     fprintf('\nSummary of maximum likelihood estimation of homogeneous cutoff rule using subsample of El-Gamal and Grether California subject pool located at %s\n',subsample);
   end
   fprintf('Highest log-likehood is %g\n',mll);
   fprintf('maximum likelihood estimate of error rate: %g  and maximum likelihood cutoff rule is:\n',err_rate_best);
   bestc

   fprintf('Highest IC value is %g and occurs for  cutoff rule:\n',mic);
   bestic

   fprintf('Total observations in all trials: %i   Total number of subjects in the trials: %i\n',total_obs,total_subjects);

   end  % end of function estimate_onetype_eg_model.m

 function [mll,bestcr,err_rate_best]=estimate_twotype_eg_model(datastruct,subsample);

 % estimate_twotype_eg_model.m  estimates the two type subcase of the El Gamal and Grether model via the EC algorithm
 %                              compare to table 2 second group of rows in their 1995 paper, "Are People Bayesian?" in JASA

 % Outputs:
 %
 % mll       maximized value of the log-likehood function
 % bestcr    optimal set of cutoff rules, i.e. the set of k cutoff rules that maximize the likelihood function 
 % err_rate  the probability of doing 50/50 random choice of the two urns, common to all subjects for each cutoff rule
 
 % note that the function [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_multitype(cr,datastruct,subsample) 
 % calculates the log likelihood (llf), information criterion and the maximum likelihood pooled error rate for cutoff rules in a matrix cr
 % which has k rows and 3 columns. Each row is a separate cutoff rule. This is done as a do-loop over all pairs of distinct 
 % cutoff rules that are "reasonable" where instead of searching over all 512 possible cutoff rules where the cutoffs are defined by triples 
 % of integers between -1 and 6 (see page 1139 of El Gamal and Grether for further explanation) this program searches over a smaller subject
 % where the cutoffs are restricted to integers between 1 and 5. The innermost call using fminbnd to calculate the maximum likelihood
 % likelihood estimate of the common error rate, err_rate, that all subjects are presumed to behave according to, after an inner most EC
 % algorithm has classified each subject according to the rule that has the highest likelihood for them given the current estimate of the error rate.

   llv=zeros(125,1);
   icv=zeros(125,1);
   errv=zeros(125,1);
   clist=[];
   i=0;

   for c1=1:5
     for c2=1:5
       for c3=1:5
           i=i+1;
           clist=[clist; [c1 c2 c3]];
       end
     end
   end

   % the order of rules does not matter and they must be distinct, so code below
   % creates an array of structures called cpairs that has 7750 distinct pairs of cutoff rules

   cpairs=struct;
   ccp=[];
   k=0;
   for i=1:125
     for j=1:125;
      if (i ~= j)
        if (numel(ccp))
          ind=find(sum(ccp == [clist(j,:)'; clist(i,:)'])==6);
          if (numel(ind) == 0)
            k=k+1;
            cpairs(k).cr=[clist(i,:); clist(j,:)];
            ccp=[ccp [clist(i,:)'; clist(j,:)']];
          end
        else
          k=k+1;
          cpairs(k).cr=[clist(i,:); clist(j,:)];
          ccp=[ccp [clist(i,:)'; clist(j,:)']];
        end
      end
     end
   end

   ncr=numel(cpairs);

   for i=1:ncr;

      [x,v]=fminbnd(@(x) -learning_how_to_learn.eg_lf_multitype(x,cpairs(i).cr,datastruct,subsample),0,1);

           err_rate=x;
           llf=-v;
           if (i == 1)
              mll=llf;
              err_rate_best=err_rate;
              bestcr=cpairs(i).cr;
           else
              if (llf > mll)
                mll=llf;
                err_rate_best=err_rate;
                bestcr=cpairs(i).cr;
              end
           end
           fprintf('%i cutoff pair (c1,c2,c3)=(%i,%i,%i) (b1,b2,b3)=(%i,%i,%i) llf=%g best likelihood so far: %g error_rate=%g\n',i,cpairs(i).cr(1,:),cpairs(i).cr(2,:),llf,mll,err_rate); 

   end

   fprintf('\nSearch completed over 7750 pairs of decision rules\n');
   fprintf('Maximized log-likelihood: %g Maximum likelihood error probability %g  Maximum likelihood cutoff pair is:\n',mll,err_rate_best);
   bestcr

  end  % end of function estimate_twotype_eg_model.m

end % methods

end % classdef
