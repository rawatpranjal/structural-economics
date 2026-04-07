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
M_.fname = 'model2';
M_.dynare_version = '5.3';
oo_.dynare_version = '5.3';
options_.dynare_version = '5.3';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(3,1);
M_.exo_names_tex = cell(3,1);
M_.exo_names_long = cell(3,1);
M_.exo_names(1) = {'eps_a'};
M_.exo_names_tex(1) = {'{\varepsilon_a}'};
M_.exo_names_long(1) = {'technology shock'};
M_.exo_names(2) = {'eps_a_news'};
M_.exo_names_tex(2) = {'{\varepsilon_a_news}'};
M_.exo_names_long(2) = {'news shock'};
M_.exo_names(3) = {'eps_z'};
M_.exo_names_tex(3) = {'{\varepsilon_z}'};
M_.exo_names_long(3) = {'preference shock'};
M_.endo_names = cell(20,1);
M_.endo_names_tex = cell(20,1);
M_.endo_names_long = cell(20,1);
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
M_.endo_names(13) = {'AUX_EXO_LAG_13_0'};
M_.endo_names_tex(13) = {'AUX\_EXO\_LAG\_13\_0'};
M_.endo_names_long(13) = {'AUX_EXO_LAG_13_0'};
M_.endo_names(14) = {'AUX_EXO_LAG_13_1'};
M_.endo_names_tex(14) = {'AUX\_EXO\_LAG\_13\_1'};
M_.endo_names_long(14) = {'AUX_EXO_LAG_13_1'};
M_.endo_names(15) = {'AUX_EXO_LAG_13_2'};
M_.endo_names_tex(15) = {'AUX\_EXO\_LAG\_13\_2'};
M_.endo_names_long(15) = {'AUX_EXO_LAG_13_2'};
M_.endo_names(16) = {'AUX_EXO_LAG_13_3'};
M_.endo_names_tex(16) = {'AUX\_EXO\_LAG\_13\_3'};
M_.endo_names_long(16) = {'AUX_EXO_LAG_13_3'};
M_.endo_names(17) = {'AUX_EXO_LAG_13_4'};
M_.endo_names_tex(17) = {'AUX\_EXO\_LAG\_13\_4'};
M_.endo_names_long(17) = {'AUX_EXO_LAG_13_4'};
M_.endo_names(18) = {'AUX_EXO_LAG_13_5'};
M_.endo_names_tex(18) = {'AUX\_EXO\_LAG\_13\_5'};
M_.endo_names_long(18) = {'AUX_EXO_LAG_13_5'};
M_.endo_names(19) = {'AUX_EXO_LAG_13_6'};
M_.endo_names_tex(19) = {'AUX\_EXO\_LAG\_13\_6'};
M_.endo_names_long(19) = {'AUX_EXO_LAG_13_6'};
M_.endo_names(20) = {'AUX_EXO_LAG_13_7'};
M_.endo_names_tex(20) = {'AUX\_EXO\_LAG\_13\_7'};
M_.endo_names_long(20) = {'AUX_EXO_LAG_13_7'};
M_.endo_partitions = struct();
M_.param_names = cell(10,1);
M_.param_names_tex = cell(10,1);
M_.param_names_long = cell(10,1);
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
M_.param_names(9) = {'PHI_PIE'};
M_.param_names_tex(9) = {'{\phi_{\pi}}'};
M_.param_names_long(9) = {'inflation feedback Taylor Rule'};
M_.param_names(10) = {'PHI_Y'};
M_.param_names_tex(10) = {'{\phi_{y}}'};
M_.param_names_long(10) = {'output feedback Taylor Rule'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 20;
M_.param_nbr = 10;
M_.orig_endo_nbr = 12;
M_.aux_vars(1).endo_index = 13;
M_.aux_vars(1).type = 3;
M_.aux_vars(1).orig_index = 2;
M_.aux_vars(1).orig_lead_lag = 0;
M_.aux_vars(1).orig_expr = 'eps_a_news';
M_.aux_vars(2).endo_index = 14;
M_.aux_vars(2).type = 3;
M_.aux_vars(2).orig_index = 2;
M_.aux_vars(2).orig_lead_lag = -1;
M_.aux_vars(2).orig_expr = 'AUX_EXO_LAG_13_0(-1)';
M_.aux_vars(3).endo_index = 15;
M_.aux_vars(3).type = 3;
M_.aux_vars(3).orig_index = 2;
M_.aux_vars(3).orig_lead_lag = -2;
M_.aux_vars(3).orig_expr = 'AUX_EXO_LAG_13_1(-1)';
M_.aux_vars(4).endo_index = 16;
M_.aux_vars(4).type = 3;
M_.aux_vars(4).orig_index = 2;
M_.aux_vars(4).orig_lead_lag = -3;
M_.aux_vars(4).orig_expr = 'AUX_EXO_LAG_13_2(-1)';
M_.aux_vars(5).endo_index = 17;
M_.aux_vars(5).type = 3;
M_.aux_vars(5).orig_index = 2;
M_.aux_vars(5).orig_lead_lag = -4;
M_.aux_vars(5).orig_expr = 'AUX_EXO_LAG_13_3(-1)';
M_.aux_vars(6).endo_index = 18;
M_.aux_vars(6).type = 3;
M_.aux_vars(6).orig_index = 2;
M_.aux_vars(6).orig_lead_lag = -5;
M_.aux_vars(6).orig_expr = 'AUX_EXO_LAG_13_4(-1)';
M_.aux_vars(7).endo_index = 19;
M_.aux_vars(7).type = 3;
M_.aux_vars(7).orig_index = 2;
M_.aux_vars(7).orig_lead_lag = -6;
M_.aux_vars(7).orig_expr = 'AUX_EXO_LAG_13_5(-1)';
M_.aux_vars(8).endo_index = 20;
M_.aux_vars(8).type = 3;
M_.aux_vars(8).orig_index = 2;
M_.aux_vars(8).orig_lead_lag = -7;
M_.aux_vars(8).orig_expr = 'AUX_EXO_LAG_13_6(-1)';
options_.varobs = cell(12, 1);
options_.varobs(1)  = {'a'};
options_.varobs(2)  = {'z'};
options_.varobs(3)  = {'c'};
options_.varobs(4)  = {'y'};
options_.varobs(5)  = {'y_nat'};
options_.varobs(6)  = {'y_gap'};
options_.varobs(7)  = {'r_nat'};
options_.varobs(8)  = {'r_real'};
options_.varobs(9)  = {'ii'};
options_.varobs(10)  = {'pie'};
options_.varobs(11)  = {'n'};
options_.varobs(12)  = {'w'};
options_.varobs_id = [ 1 2 3 4 5 6 7 8 9 10 11 12  ];
M_ = setup_solvers(M_);
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
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
M_.eq_nbr = 20;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 1;
M_.orig_maximum_exo_lag = 8;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 8;
M_.orig_maximum_lead = 1;
M_.orig_maximum_lag_with_diffs_expanded = 8;
M_.lead_lag_incidence = [
 1 11 0;
 2 12 0;
 0 13 0;
 0 14 0;
 0 15 0;
 0 16 31;
 0 17 0;
 0 18 0;
 0 19 0;
 0 20 32;
 0 21 0;
 0 22 0;
 3 23 0;
 4 24 0;
 5 25 0;
 6 26 0;
 7 27 0;
 8 28 0;
 9 29 0;
 10 30 0;]';
M_.nstatic = 8;
M_.nfwrd   = 2;
M_.npred   = 10;
M_.nboth   = 0;
M_.nsfwrd   = 2;
M_.nspred   = 10;
M_.ndynamic   = 12;
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
  12 , 'name' , 'ii' ;
};
M_.mapping.a.eqidx = [3 6 8 10 ];
M_.mapping.z.eqidx = [7 8 ];
M_.mapping.c.eqidx = [4 5 ];
M_.mapping.y.eqidx = [3 5 11 ];
M_.mapping.y_nat.eqidx = [10 11 ];
M_.mapping.y_gap.eqidx = [1 2 11 12 ];
M_.mapping.r_nat.eqidx = [2 8 12 ];
M_.mapping.r_real.eqidx = [9 ];
M_.mapping.ii.eqidx = [2 9 12 ];
M_.mapping.pie.eqidx = [1 2 9 12 ];
M_.mapping.n.eqidx = [3 4 ];
M_.mapping.w.eqidx = [4 ];
M_.mapping.eps_a.eqidx = [6 ];
M_.mapping.eps_a_news.eqidx = [6 ];
M_.mapping.eps_z.eqidx = [7 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [1 2 13 14 15 16 17 18 19 20 ];
M_.exo_names_orig_ord = [1:3];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(20, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(10, 1);
M_.endo_trends = struct('deflator', cell(20, 1), 'log_deflator', cell(20, 1), 'growth_factor', cell(20, 1), 'log_growth_factor', cell(20, 1));
M_.NNZDerivatives = [54; 0; -1; ];
M_.static_tmp_nbr = [3; 0; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
M_.Sigma_e(2, 2) = 1;
M_.Sigma_e(3, 3) = 1;
M_.params(5) = 1;
SIGMA = M_.params(5);
M_.params(6) = 5;
VARPHI = M_.params(6);
M_.params(8) = 0.75;
THETA = M_.params(8);
M_.params(4) = 0.5;
RHOZ = M_.params(4);
M_.params(3) = 0.9;
RHOA = M_.params(3);
M_.params(2) = 0.99;
BETA = M_.params(2);
M_.params(1) = 0.25;
ALPHA = M_.params(1);
M_.params(7) = 9;
EPSILON = M_.params(7);
M_.params(9) = 1.5;
PHI_PIE = M_.params(9);
M_.params(10) = 0.125;
PHI_Y = M_.params(10);
steady;
oo_.dr.eigval = check(M_,options_,oo_);
options_.order = 2;
var_list_ = {};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);
estim_params_.var_exo = zeros(0, 10);
estim_params_.var_endo = zeros(0, 10);
estim_params_.corrx = zeros(0, 11);
estim_params_.corrn = zeros(0, 11);
estim_params_.param_vals = zeros(0, 10);
estim_params_.param_vals = [estim_params_.param_vals; 9, 1.5, 0, 2, 0, NaN, NaN, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 10, 0.125, 0, 2, 0, NaN, NaN, NaN, NaN, NaN ];


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'model2_results.mat'], 'oo_recursive_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
