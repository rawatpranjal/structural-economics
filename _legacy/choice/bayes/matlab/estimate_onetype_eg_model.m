% estimate_onetype_eg_model.m  estimates the single type subcase of the El Gamal and Grether model,
%                              compare to table 2 top rows in their 1995 paper, "Are People Bayesian?" in JASA

% note that the function [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_singletype([c1 c2 c3],datastruct,subsample) 
% calculates the log likelihood (llf), information criterion and the maximum likelihood pooled error rate for cutoff rule [c1 c2 c3]
% where these are integers between -1 and 6 (see page 1139 of El Gamal and Grether for further explanation). The likelihood
% routine returns the maximum likelihood estimate of the common error rate, err_rat, that all subbjects are presumed to behave
% according to, in conjunction with the cutoff rule [c1 c2 c3] where the 3 values index the 3 possible priors (1/3,1/2,2/3) used
% in the experiment. This program does a more localized discrete search for cutoffs from 1 to 5 and does it as a brute force
% search in a simple do-loop.

   if (~exist('datastruct'))
     load('datastruct');
   end

   subsample='OXY';

   llv=zeros(125,1);
   icv=zeros(125,1);
   errv=zeros(125,1);
   i=0;

   for c1=1:5
     for c2=1:5
       for c3=1:5
           i=i+1;
           [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_singletype([c1 c2 c3],datastruct,subsample);
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
