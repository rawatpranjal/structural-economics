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
M_.fname = 'model';
M_.dynare_version = '5.3';
oo_.dynare_version = '5.3';
options_.dynare_version = '5.3';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(2,1);
M_.exo_names_tex = cell(2,1);
M_.exo_names_long = cell(2,1);
M_.exo_names(1) = {'eps_a'};
M_.exo_names_tex(1) = {'{\varepsilon_a}'};
M_.exo_names_long(1) = {'technology shock'};
M_.exo_names(2) = {'eps_z'};
M_.exo_names_tex(2) = {'{\varepsilon_z}'};
M_.exo_names_long(2) = {'preference shock'};
M_.endo_names = cell(12,1);
M_.endo_names_tex = cell(12,1);
M_.endo_names_long = cell(12,1);
M_.endo_names(1) = {'a'};
M_.endo_names_tex(1) = {'{a}'};
M_.endo_names_long(1) = {'technology shock process (log dev ss)'};
M_.endo_names(2) = {'z'};
M_.endo_names_tex(2) = {'{z}'};
M_.endo_names_long(2) = {'preference shock process (log dev ss)'};
M_.endo_names(3) = {'c'};
M_.endo_names_tex(3) = {'{c}'};
M_.endo_names_long(3) = {'consumption (log dev ss)'};
M_.endo_names(4) = {'y'};
M_.endo_names_tex(4) = {'{y}'};
M_.endo_names_long(4) = {'output (log dev ss)'};
M_.endo_names(5) = {'y_nat'};
M_.endo_names_tex(5) = {'{y^{nat}}'};
M_.endo_names_long(5) = {'natural output (log dev ss)'};
M_.endo_names(6) = {'y_gap'};
M_.endo_names_tex(6) = {'{\tilde y}'};
M_.endo_names_long(6) = {'output gap (log dev ss)'};
M_.endo_names(7) = {'r_nat'};
M_.endo_names_tex(7) = {'{r^{nat}}'};
M_.endo_names_long(7) = {'natural interest rate (log dev ss)'};
M_.endo_names(8) = {'r_real'};
M_.endo_names_tex(8) = {'{r}'};
M_.endo_names_long(8) = {'real interest rate (log dev ss)'};
M_.endo_names(9) = {'ii'};
M_.endo_names_tex(9) = {'{i}'};
M_.endo_names_long(9) = {'nominal interest rate (log dev ss)'};
M_.endo_names(10) = {'pie'};
M_.endo_names_tex(10) = {'{\pi}'};
M_.endo_names_long(10) = {'inflation (log dev ss)'};
M_.endo_names(11) = {'n'};
M_.endo_names_tex(11) = {'{n}'};
M_.endo_names_long(11) = {'hours worked (log dev ss)'};
M_.endo_names(12) = {'w'};
M_.endo_names_tex(12) = {'{w}'};
M_.endo_names_long(12) = {'real wage (log dev ss)'};
M_.endo_partitions = struct();
M_.param_names = cell(8,1);
M_.param_names_tex = cell(8,1);
M_.param_names_long = cell(8,1);
M_.param_names(1) = {'ALPHA'};
M_.param_names_tex(1) = {'{\alpha}'};
M_.param_names_long(1) = {'one minus labor share in production'};
M_.param_names(2) = {'BETA'};
M_.param_names_tex(2) = {'{\beta}'};
M_.param_names_long(2) = {'discount factor'};
M_.param_names(3) = {'RHOA'};
M_.param_names_tex(3) = {'{\rho_a}'};
M_.param_names_long(3) = {'autocorrelation technology process'};
M_.param_names(4) = {'RHOZ'};
M_.param_names_tex(4) = {'{\rho_{z}}'};
M_.param_names_long(4) = {'autocorrelation preference process'};
M_.param_names(5) = {'SIGMA'};
M_.param_names_tex(5) = {'{\sigma}'};
M_.param_names_long(5) = {'inverse EIS'};
M_.param_names(6) = {'VARPHI'};
M_.param_names_tex(6) = {'{\varphi}'};
M_.param_names_long(6) = {'inverse Frisch elasticity'};
M_.param_names(7) = {'EPSILON'};
M_.param_names_tex(7) = {'{\epsilon}'};
M_.param_names_long(7) = {'Dixit-Stiglitz demand elasticity'};
M_.param_names(8) = {'THETA'};
M_.param_names_tex(8) = {'{\theta}'};
M_.param_names_long(8) = {'Calvo probability'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 2;
M_.endo_nbr = 12;
M_.param_nbr = 8;
M_.orig_endo_nbr = 12;
M_.aux_vars = [];
M_ = setup_solvers(M_);
M_.Sigma_e = zeros(2, 2);
M_.Correlation_matrix = eye(2, 2);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
options_.linear = true;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
M_.nonzero_hessian_eqs = [];
M_.hessian_eq_zero = isempty(M_.nonzero_hessian_eqs);
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
 1 3 0;
 2 4 0;
 0 5 0;
 0 6 0;
 0 7 0;
 0 8 15;
 0 9 0;
 0 10 0;
 0 11 0;
 0 12 16;
 0 13 0;
 0 14 0;]';
M_.nstatic = 8;
M_.nfwrd   = 2;
M_.npred   = 2;
M_.nboth   = 0;
M_.nsfwrd   = 2;
M_.nspred   = 2;
M_.ndynamic   = 4;
M_.dynamic_tmp_nbr = [3; 0; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , 'New Keynesian Phillips Curve' ;
  2 , 'name' , 'Dynamic IS Curve' ;
  3 , 'name' , 'Production function' ;
  4 , 'name' , 'labor demand' ;
  5 , 'name' , 'resource constraint' ;
  6 , 'name' , 'TFP process' ;
  7 , 'name' , 'Preference shifter' ;
  8 , 'name' , 'Definition natural rate of interest' ;
  9 , 'name' , 'Definition real interest rate' ;
  10 , 'name' , 'Definition natural output' ;
  11 , 'name' , 'Definition output gap' ;
  12 , 'name' , 'Interest Rate Rule: Exogenous One-To-One' ;
};
M_.mapping.a.eqidx = [3 6 8 10 ];
M_.mapping.z.eqidx = [7 8 ];
M_.mapping.c.eqidx = [4 5 ];
M_.mapping.y.eqidx = [3 5 11 ];
M_.mapping.y_nat.eqidx = [10 11 ];
M_.mapping.y_gap.eqidx = [1 2 11 ];
M_.mapping.r_nat.eqidx = [2 8 12 ];
M_.mapping.r_real.eqidx = [9 ];
M_.mapping.ii.eqidx = [2 9 12 ];
M_.mapping.pie.eqidx = [1 2 9 ];
M_.mapping.n.eqidx = [3 4 ];
M_.mapping.w.eqidx = [4 ];
M_.mapping.eps_a.eqidx = [6 ];
M_.mapping.eps_z.eqidx = [7 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [1 2 ];
M_.exo_names_orig_ord = [1:2];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(12, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(2, 1);
M_.params = NaN(8, 1);
M_.endo_trends = struct('deflator', cell(12, 1), 'log_deflator', cell(12, 1), 'growth_factor', cell(12, 1), 'log_growth_factor', cell(12, 1));
M_.NNZDerivatives = [35; 0; -1; ];
M_.static_tmp_nbr = [3; 0; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
M_.Sigma_e(2, 2) = 1;
options_.order = 2;
var_list_ = {};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model_results.mat'], 'oo_recursive_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
