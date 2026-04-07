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
M_.exo_names(1) = {'eps_z_news'};
M_.exo_names_tex(1) = {'{\varepsilon_z^{news}}'};
M_.exo_names_long(1) = {'eps_z_news'};
M_.exo_names(2) = {'eps_z_surprise'};
M_.exo_names_tex(2) = {'{\varepsilon_z^{surprise}}'};
M_.exo_names_long(2) = {'eps_z_surprise'};
M_.endo_names = cell(16,1);
M_.endo_names_tex = cell(16,1);
M_.endo_names_long = cell(16,1);
M_.endo_names(1) = {'y'};
M_.endo_names_tex(1) = {'y'};
M_.endo_names_long(1) = {'y'};
M_.endo_names(2) = {'c'};
M_.endo_names_tex(2) = {'c'};
M_.endo_names_long(2) = {'c'};
M_.endo_names(3) = {'k'};
M_.endo_names_tex(3) = {'k'};
M_.endo_names_long(3) = {'k'};
M_.endo_names(4) = {'l'};
M_.endo_names_tex(4) = {'l'};
M_.endo_names_long(4) = {'l'};
M_.endo_names(5) = {'z'};
M_.endo_names_tex(5) = {'z'};
M_.endo_names_long(5) = {'z'};
M_.endo_names(6) = {'r'};
M_.endo_names_tex(6) = {'r'};
M_.endo_names_long(6) = {'r'};
M_.endo_names(7) = {'w'};
M_.endo_names_tex(7) = {'w'};
M_.endo_names_long(7) = {'w'};
M_.endo_names(8) = {'invest'};
M_.endo_names_tex(8) = {'{i}'};
M_.endo_names_long(8) = {'invest'};
M_.endo_names(9) = {'AUX_EXO_LAG_8_0'};
M_.endo_names_tex(9) = {'AUX\_EXO\_LAG\_8\_0'};
M_.endo_names_long(9) = {'AUX_EXO_LAG_8_0'};
M_.endo_names(10) = {'AUX_EXO_LAG_8_1'};
M_.endo_names_tex(10) = {'AUX\_EXO\_LAG\_8\_1'};
M_.endo_names_long(10) = {'AUX_EXO_LAG_8_1'};
M_.endo_names(11) = {'AUX_EXO_LAG_8_2'};
M_.endo_names_tex(11) = {'AUX\_EXO\_LAG\_8\_2'};
M_.endo_names_long(11) = {'AUX_EXO_LAG_8_2'};
M_.endo_names(12) = {'AUX_EXO_LAG_8_3'};
M_.endo_names_tex(12) = {'AUX\_EXO\_LAG\_8\_3'};
M_.endo_names_long(12) = {'AUX_EXO_LAG_8_3'};
M_.endo_names(13) = {'AUX_EXO_LAG_8_4'};
M_.endo_names_tex(13) = {'AUX\_EXO\_LAG\_8\_4'};
M_.endo_names_long(13) = {'AUX_EXO_LAG_8_4'};
M_.endo_names(14) = {'AUX_EXO_LAG_8_5'};
M_.endo_names_tex(14) = {'AUX\_EXO\_LAG\_8\_5'};
M_.endo_names_long(14) = {'AUX_EXO_LAG_8_5'};
M_.endo_names(15) = {'AUX_EXO_LAG_8_6'};
M_.endo_names_tex(15) = {'AUX\_EXO\_LAG\_8\_6'};
M_.endo_names_long(15) = {'AUX_EXO_LAG_8_6'};
M_.endo_names(16) = {'AUX_EXO_LAG_8_7'};
M_.endo_names_tex(16) = {'AUX\_EXO\_LAG\_8\_7'};
M_.endo_names_long(16) = {'AUX_EXO_LAG_8_7'};
M_.endo_partitions = struct();
M_.param_names = cell(11,1);
M_.param_names_tex = cell(11,1);
M_.param_names_long = cell(11,1);
M_.param_names(1) = {'beta'};
M_.param_names_tex(1) = {'\beta'};
M_.param_names_long(1) = {'beta'};
M_.param_names(2) = {'psi'};
M_.param_names_tex(2) = {'\psi'};
M_.param_names_long(2) = {'psi'};
M_.param_names(3) = {'sigma'};
M_.param_names_tex(3) = {'\sigma'};
M_.param_names_long(3) = {'sigma'};
M_.param_names(4) = {'delta'};
M_.param_names_tex(4) = {'\delta'};
M_.param_names_long(4) = {'delta'};
M_.param_names(5) = {'alpha'};
M_.param_names_tex(5) = {'\alpha'};
M_.param_names_long(5) = {'alpha'};
M_.param_names(6) = {'rhoz'};
M_.param_names_tex(6) = {'\rho_z'};
M_.param_names_long(6) = {'rhoz'};
M_.param_names(7) = {'gammax'};
M_.param_names_tex(7) = {'\gamma_x'};
M_.param_names_long(7) = {'gammax'};
M_.param_names(8) = {'n'};
M_.param_names_tex(8) = {'n'};
M_.param_names_long(8) = {'n'};
M_.param_names(9) = {'x'};
M_.param_names_tex(9) = {'x'};
M_.param_names_long(9) = {'x'};
M_.param_names(10) = {'i_y'};
M_.param_names_tex(10) = {'i\_y'};
M_.param_names_long(10) = {'i_y'};
M_.param_names(11) = {'k_y'};
M_.param_names_tex(11) = {'k\_y'};
M_.param_names_long(11) = {'k_y'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 2;
M_.endo_nbr = 16;
M_.param_nbr = 11;
M_.orig_endo_nbr = 8;
M_.aux_vars(1).endo_index = 9;
M_.aux_vars(1).type = 3;
M_.aux_vars(1).orig_index = 1;
M_.aux_vars(1).orig_lead_lag = 0;
M_.aux_vars(1).orig_expr = 'eps_z_news';
M_.aux_vars(2).endo_index = 10;
M_.aux_vars(2).type = 3;
M_.aux_vars(2).orig_index = 1;
M_.aux_vars(2).orig_lead_lag = -1;
M_.aux_vars(2).orig_expr = 'AUX_EXO_LAG_8_0(-1)';
M_.aux_vars(3).endo_index = 11;
M_.aux_vars(3).type = 3;
M_.aux_vars(3).orig_index = 1;
M_.aux_vars(3).orig_lead_lag = -2;
M_.aux_vars(3).orig_expr = 'AUX_EXO_LAG_8_1(-1)';
M_.aux_vars(4).endo_index = 12;
M_.aux_vars(4).type = 3;
M_.aux_vars(4).orig_index = 1;
M_.aux_vars(4).orig_lead_lag = -3;
M_.aux_vars(4).orig_expr = 'AUX_EXO_LAG_8_2(-1)';
M_.aux_vars(5).endo_index = 13;
M_.aux_vars(5).type = 3;
M_.aux_vars(5).orig_index = 1;
M_.aux_vars(5).orig_lead_lag = -4;
M_.aux_vars(5).orig_expr = 'AUX_EXO_LAG_8_3(-1)';
M_.aux_vars(6).endo_index = 14;
M_.aux_vars(6).type = 3;
M_.aux_vars(6).orig_index = 1;
M_.aux_vars(6).orig_lead_lag = -5;
M_.aux_vars(6).orig_expr = 'AUX_EXO_LAG_8_4(-1)';
M_.aux_vars(7).endo_index = 15;
M_.aux_vars(7).type = 3;
M_.aux_vars(7).orig_index = 1;
M_.aux_vars(7).orig_lead_lag = -6;
M_.aux_vars(7).orig_expr = 'AUX_EXO_LAG_8_5(-1)';
M_.aux_vars(8).endo_index = 16;
M_.aux_vars(8).type = 3;
M_.aux_vars(8).orig_index = 1;
M_.aux_vars(8).orig_lead_lag = -7;
M_.aux_vars(8).orig_expr = 'AUX_EXO_LAG_8_6(-1)';
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
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
M_.orig_eq_nbr = 8;
M_.eq_nbr = 16;
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
 0 11 0;
 0 12 27;
 1 13 0;
 0 14 28;
 2 15 29;
 0 16 0;
 0 17 0;
 0 18 0;
 3 19 0;
 4 20 0;
 5 21 0;
 6 22 0;
 7 23 0;
 8 24 0;
 9 25 0;
 10 26 0;]';
M_.nstatic = 4;
M_.nfwrd   = 2;
M_.npred   = 9;
M_.nboth   = 1;
M_.nsfwrd   = 3;
M_.nspred   = 10;
M_.ndynamic   = 12;
M_.dynamic_tmp_nbr = [7; 1; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , '1' ;
  2 , 'name' , '2' ;
  3 , 'name' , '3' ;
  4 , 'name' , '4' ;
  5 , 'name' , '5' ;
  6 , 'name' , '6' ;
  7 , 'name' , 'r' ;
  8 , 'name' , 'z' ;
};
M_.mapping.y.eqidx = [4 5 6 7 ];
M_.mapping.c.eqidx = [1 2 4 ];
M_.mapping.k.eqidx = [1 3 5 7 ];
M_.mapping.l.eqidx = [1 2 5 6 ];
M_.mapping.z.eqidx = [1 5 8 ];
M_.mapping.r.eqidx = [7 ];
M_.mapping.w.eqidx = [2 6 ];
M_.mapping.invest.eqidx = [3 4 ];
M_.mapping.eps_z_news.eqidx = [8 ];
M_.mapping.eps_z_surprise.eqidx = [8 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [3 5 9 10 11 12 13 14 15 16 ];
M_.exo_names_orig_ord = [1:2];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(16, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(2, 1);
M_.params = NaN(11, 1);
M_.endo_trends = struct('deflator', cell(16, 1), 'log_deflator', cell(16, 1), 'growth_factor', cell(16, 1), 'log_growth_factor', cell(16, 1));
M_.NNZDerivatives = [44; -1; -1; ];
M_.static_tmp_nbr = [8; 2; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(3) = 1;
sigma = M_.params(3);
M_.params(5) = 0.33;
alpha = M_.params(5);
M_.params(10) = 0.25;
i_y = M_.params(10);
M_.params(11) = 10.4;
k_y = M_.params(11);
M_.params(9) = 0.0055;
x = M_.params(9);
M_.params(8) = 0.0027;
n = M_.params(8);
M_.params(6) = 0.97;
rhoz = M_.params(6);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
M_.Sigma_e(2, 2) = 1;
steady;
oo_.dr.eigval = check(M_,options_,oo_);
options_.irf = 40;
options_.order = 1;
var_list_ = {};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);
initial_condition_states = repmat(oo_.dr.ys,1,M_.maximum_lag);
shock_matrix = zeros(options_.irf,M_.exo_nbr); 
shock_matrix(1,strmatch('eps_z_news',M_.exo_names,'exact')) = 1; 
shock_matrix(1+8,strmatch('eps_z_surprise',M_.exo_names,'exact')) = -1; 
y2 = simult_(M_,options_,initial_condition_states,oo_.dr,shock_matrix,1);
y_IRF = y2(:,M_.maximum_lag+1:end)-repmat(oo_.dr.ys,1,options_.irf); 
figure
subplot(2,1,1)
plot(y_IRF(strmatch('y',M_.endo_names,'exact'),:)); 
title('Output');
subplot(2,1,2)
plot(y_IRF(strmatch('z',M_.endo_names,'exact'),:));
title('TFP');
figure
for ii=1:M_.orig_endo_nbr
subplot(3,3,ii)
if max(abs(y_IRF(ii,:)))>1e-12 
plot(y_IRF(ii,:));
else
plot(zeros(options_.irf,1));
end
title(deblank(M_.endo_names(ii,:)));
end


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
