% read_data.m: program to read in the experimental data from the paper "Are People Bayesian?" by El-Gamal and Grether
%              John Rust, Gerogetown University, February 2022
%
% NOTE: we omit file PCC.pay since there seems to be issues with this file and it does not match the format of the other files

filenames={'CSULA.pay','CSULANO.pay','OXY.pay','OXYNO.pay','PCCNO.pay','UCLA.pay','UCLANO.pay'};

n=size(filenames,2);

for i=1:n

 fprintf('Processing %s\n',filenames{i});
 datastruct(i).name=filenames{i};
 filehandle=fopen(filenames{i});
 linecount=0;
 line=0;
 subjectchoices=[];
 while (line ~= -1)
   line=fgetl(filehandle);
   linecount=linecount+1;
   fprintf('processing line %i\n',linecount);
   if (linecount == 1)
     priors=sscanf(line,'%f');
     fprintf('size of priors: %i\n',numel(priors));
     datastruct(i).priors=priors;
   elseif (linecount == 2)
     ndraws=sscanf(line,'%f');
     fprintf('size of ndraws: %i\n',numel(ndraws));
     datastruct(i).ndraws=ndraws;
   elseif (isempty(line))
     fprintf('skipping blank line');
     line=0;
   elseif (line ~= -1)
     data=sscanf(line,'%f');
     ntrials=size(priors,1);
     if (size(data,1) ~= ntrials)
        fprintf('warning: subject data has %i columns, but there were %i trials performed: discarding first column\n',size(data,1),ntrials);
        subjectchoices=[subjectchoices; data(2:end)'];
     else
        subjectchoices=[subjectchoices; data'];
     end
   else
     fprintf('end of file reached, terminating scan\n');
     datastruct(i).subjectchoices=subjectchoices;
     break;
   end
 end
 fclose(filehandle);

end

% add a pay field to specify whether subjects were paid in the experiment for guessing the right cage

datastruct(1).pay=1;
datastruct(2).pay=0;
datastruct(3).pay=1;
datastruct(4).pay=0;
datastruct(5).pay=0;
datastruct(6).pay=1;
datastruct(7).pay=0;

datastruct(1).nballs_prior_cage=6;
datastruct(2).nballs_prior_cage=6;
datastruct(3).nballs_prior_cage=6;
datastruct(4).nballs_prior_cage=6;
datastruct(5).nballs_prior_cage=6;
datastruct(6).nballs_prior_cage=6;
datastruct(7).nballs_prior_cage=6;

save('datastruct','datastruct');

