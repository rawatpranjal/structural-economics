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

function [llf,ic,total_obs,total_subjects,err_rate]=eg_lf_singletype(cr,datastruct,school)

    numexps=numel(datastruct);
    llf=0;
    llf_homo=0;
    total_consistent=0;
    total_obs=0;
    total_subjects=0;

    k=size(cr,1);

    for e=1:numexps;    % loop over all experiments to calculate likelihood
      if (startsWith(datastruct(e).name,school))    % if the school string is provided, then likelihood is calculated for given school

        nsubjects=size(datastruct(e).subjectchoices,1);
        priors=datastruct(e).priors;
        ntrials=numel(priors);
        ndraws=datastruct(e).ndraws;

        cr_prior=cr(priors-1)';  % trial specific cutoff depending on the prior in that trails
        cr_choice=(ndraws > cr_prior)';  % predicted choices for each trial implied by the cutoff rule
        consistent_obs=sum((cr_choice == datastruct(e).subjectchoices)');
        err_rates=2*(1-sum(consistent_obs)/(nsubjects*ntrials));

        total_subjects=total_subjects+nsubjects;
        total_consistent=total_consistent+sum(consistent_obs);
        total_obs=total_obs+nsubjects*ntrials;

        llf=llf+sum(consistent_obs.*log(1-err_rates/2)+(ntrials-consistent_obs).*log(err_rates/2));
      end
    end

    err_rate=2*(1-total_consistent/total_obs);

    llf=total_consistent*log(1-err_rates/2)+(total_obs-total_consistent)*log(err_rates/2);

    ic=llf-3*k*log(8)-k*log(2)-total_subjects*log(k);
      
end 
