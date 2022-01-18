%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'ClassicalMonetaryModel';
M_.dynare_version = '4.6.3';
oo_.dynare_version = '4.6.3';
options_.dynare_version = '4.6.3';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('ClassicalMonetaryModel.log');
M_.exo_names = cell(3,1);
M_.exo_names_tex = cell(3,1);
M_.exo_names_long = cell(3,1);
M_.exo_names(1) = {'eps_a'};
M_.exo_names_tex(1) = {'eps\_a'};
M_.exo_names_long(1) = {'eps_a'};
M_.exo_names(2) = {'eps_z'};
M_.exo_names_tex(2) = {'eps\_z'};
M_.exo_names_long(2) = {'eps_z'};
M_.exo_names(3) = {'eps_v'};
M_.exo_names_tex(3) = {'eps\_v'};
M_.exo_names_long(3) = {'eps_v'};
M_.endo_names = cell(12,1);
M_.endo_names_tex = cell(12,1);
M_.endo_names_long = cell(12,1);
M_.endo_names(1) = {'C'};
M_.endo_names_tex(1) = {'C'};
M_.endo_names_long(1) = {'C'};
M_.endo_names(2) = {'W'};
M_.endo_names_tex(2) = {'W'};
M_.endo_names_long(2) = {'W'};
M_.endo_names(3) = {'Pi'};
M_.endo_names_tex(3) = {'Pi'};
M_.endo_names_long(3) = {'Pi'};
M_.endo_names(4) = {'A'};
M_.endo_names_tex(4) = {'A'};
M_.endo_names_long(4) = {'A'};
M_.endo_names(5) = {'N'};
M_.endo_names_tex(5) = {'N'};
M_.endo_names_long(5) = {'N'};
M_.endo_names(6) = {'i'};
M_.endo_names_tex(6) = {'i'};
M_.endo_names_long(6) = {'i'};
M_.endo_names(7) = {'R'};
M_.endo_names_tex(7) = {'R'};
M_.endo_names_long(7) = {'R'};
M_.endo_names(8) = {'Y'};
M_.endo_names_tex(8) = {'Y'};
M_.endo_names_long(8) = {'Y'};
M_.endo_names(9) = {'V'};
M_.endo_names_tex(9) = {'V'};
M_.endo_names_long(9) = {'V'};
M_.endo_names(10) = {'dM'};
M_.endo_names_tex(10) = {'dM'};
M_.endo_names_long(10) = {'dM'};
M_.endo_names(11) = {'Q'};
M_.endo_names_tex(11) = {'Q'};
M_.endo_names_long(11) = {'Q'};
M_.endo_names(12) = {'Z'};
M_.endo_names_tex(12) = {'Z'};
M_.endo_names_long(12) = {'Z'};
M_.endo_partitions = struct();
M_.param_names = cell(9,1);
M_.param_names_tex = cell(9,1);
M_.param_names_long = cell(9,1);
M_.param_names(1) = {'alpha'};
M_.param_names_tex(1) = {'alpha'};
M_.param_names_long(1) = {'alpha'};
M_.param_names(2) = {'beta'};
M_.param_names_tex(2) = {'beta'};
M_.param_names_long(2) = {'beta'};
M_.param_names(3) = {'sigma'};
M_.param_names_tex(3) = {'sigma'};
M_.param_names_long(3) = {'sigma'};
M_.param_names(4) = {'gamma'};
M_.param_names_tex(4) = {'gamma'};
M_.param_names_long(4) = {'gamma'};
M_.param_names(5) = {'phi'};
M_.param_names_tex(5) = {'phi'};
M_.param_names_long(5) = {'phi'};
M_.param_names(6) = {'eta'};
M_.param_names_tex(6) = {'eta'};
M_.param_names_long(6) = {'eta'};
M_.param_names(7) = {'rho_a'};
M_.param_names_tex(7) = {'rho\_a'};
M_.param_names_long(7) = {'rho_a'};
M_.param_names(8) = {'rho_z'};
M_.param_names_tex(8) = {'rho\_z'};
M_.param_names_long(8) = {'rho_z'};
M_.param_names(9) = {'rho_v'};
M_.param_names_tex(9) = {'rho\_v'};
M_.param_names_long(9) = {'rho_v'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 12;
M_.param_nbr = 9;
M_.orig_endo_nbr = 12;
M_.aux_vars = [];
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
options_.linear_decomposition = false;
M_.orig_eq_nbr = 12;
M_.eq_nbr = 12;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 1;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 1;
M_.orig_maximum_lead = 1;
M_.orig_maximum_lag_with_diffs_expanded = 1;
M_.lead_lag_incidence = [
 1 6 18;
 0 7 0;
 0 8 19;
 2 9 0;
 0 10 0;
 3 11 0;
 0 12 0;
 0 13 0;
 4 14 0;
 0 15 0;
 0 16 0;
 5 17 20;]';
M_.nstatic = 6;
M_.nfwrd   = 1;
M_.npred   = 3;
M_.nboth   = 2;
M_.nsfwrd   = 3;
M_.nspred   = 5;
M_.ndynamic   = 6;
M_.dynamic_tmp_nbr = [7; 1; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , 'W' ;
  2 , 'name' , 'Q' ;
  3 , 'name' , 'i' ;
  4 , 'name' , 'Y' ;
  5 , 'name' , '5' ;
  6 , 'name' , '6' ;
  7 , 'name' , 'R' ;
  8 , 'name' , 'C' ;
  9 , 'name' , '9' ;
  10 , 'name' , '10' ;
  11 , 'name' , 'V' ;
  12 , 'name' , 'dM' ;
};
M_.mapping.C.eqidx = [1 2 8 12 ];
M_.mapping.W.eqidx = [1 5 ];
M_.mapping.Pi.eqidx = [2 6 7 12 ];
M_.mapping.A.eqidx = [4 5 9 ];
M_.mapping.N.eqidx = [1 4 5 ];
M_.mapping.i.eqidx = [3 6 12 ];
M_.mapping.R.eqidx = [6 7 ];
M_.mapping.Y.eqidx = [4 8 ];
M_.mapping.V.eqidx = [7 11 ];
M_.mapping.dM.eqidx = [12 ];
M_.mapping.Q.eqidx = [2 3 ];
M_.mapping.Z.eqidx = [2 10 ];
M_.mapping.eps_a.eqidx = [9 ];
M_.mapping.eps_z.eqidx = [10 ];
M_.mapping.eps_v.eqidx = [11 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [1 4 6 9 12 ];
M_.exo_names_orig_ord = [1:3];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(12, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(9, 1);
M_.endo_trends = struct('deflator', cell(12, 1), 'log_deflator', cell(12, 1), 'growth_factor', cell(12, 1), 'log_growth_factor', cell(12, 1));
M_.NNZDerivatives = [40; -1; -1; ];
M_.static_tmp_nbr = [5; 0; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(1) = 0.25;
alpha = M_.params(1);
M_.params(2) = 0.99;
beta = M_.params(2);
M_.params(7) = 0.9;
rho_a = M_.params(7);
M_.params(8) = 0.5;
rho_z = M_.params(8);
M_.params(9) = 0.5;
rho_v = M_.params(9);
M_.params(3) = 1;
sigma = M_.params(3);
M_.params(5) = 5;
phi = M_.params(5);
M_.params(4) = 1.5;
gamma = M_.params(4);
M_.params(6) = 3.77;
eta = M_.params(6);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = (1)^2;
M_.Sigma_e(2, 2) = (1)^2;
M_.Sigma_e(3, 3) = (1)^2;
resid(1);
steady;
oo_.dr.eigval = check(M_,options_,oo_);
options_.irf = 20;
options_.order = 1;
var_list_ = {'Y';'C';'Pi';'i';'R';'dM'};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);
save('ClassicalMonetaryModel_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save('ClassicalMonetaryModel_results.mat', 'oo_recursive_', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc(tic0)) ]);
disp('Note: 1 warning(s) encountered in the preprocessor')
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off
