% llf_deriv_checker.m  numerical check of analytical gradient and hessian of binary logit log-likelihood function
%                      John Rust, Georgetown University February, 2021 

ndelta=1e-5;  % numerical delta for taking double sided difference approximations for superaccurate
              % numerical derivatives 

dependent_variable='true_classification';
model='prior_linearly';
model='llr_plr';
errspec='logit';  % specification for the error term: 'logit' or 'probit'

%generate data for the likeihood function

training_sample_size=400;
[ydata,xdata]=learning_how_to_learn.generate_training_data(training_sample_size,dependent_variable);

if (strcmp(model,'llr_lpr'))
   truetheta=[0; -1; -1];
else
   truetheta=[6*(log(1/2)-log(1/3)); -log(2); 1];
end

if (strcmp(errspec,'probit'))
[llf,dllf,hllf]=learning_how_to_learn.bprobit(ydata,xdata,truetheta,model);
else
[llf,dllf,hllf]=learning_how_to_learn.blogit(ydata,xdata,truetheta,model);
end

fprintf('checking derivatives of log-likelihood for binary probit model using %i randomly generated training observations\n',training_sample_size);
fprintf('dependent_variable is %s and model is %s\n',dependent_variable,model);

nparms=size(truetheta,1);

ndllf=zeros(1,nparms);
nhllf=zeros(nparms,nparms);

for i=1:nparms

  theta_u=truetheta;
  theta_u(i)=theta_u(i)+ndelta;
  if (strcmp(errspec,'probit'))
  [llfu,dllfu]=learning_how_to_learn.bprobit(ydata,xdata,theta_u,model);
  else
  [llfu,dllfu]=learning_how_to_learn.blogit(ydata,xdata,theta_u,model);
  end

  theta_l=truetheta;
  theta_l(i)=theta_l(i)-ndelta;
  if (strcmp(errspec,'probit'))
  [llfl,dllfl]=learning_how_to_learn.bprobit(ydata,xdata,theta_l,model);
  else
  [llfl,dllfl]=learning_how_to_learn.blogit(ydata,xdata,theta_l,model);
  end

  ndllf(i)=(llfu-llfl)/(2*ndelta);
  nhllf(i,:)=(dllfu-dllfl)/(2*ndelta);

end

fprintf('Parameters\n');
truetheta'
fprintf('Analytical gradients\n');
dllf
fprintf('Numerical gradients\n');
ndllf
fprintf('Difference\n');
ndllf-dllf
fprintf('Analytical hessian\n');
hllf
fprintf('Numerical gradients\n');
nhllf
fprintf('Difference\n');
nhllf-hllf
