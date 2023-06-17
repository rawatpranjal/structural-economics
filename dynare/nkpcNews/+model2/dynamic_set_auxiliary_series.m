function ds = dynamic_set_auxiliary_series(ds, params)
%
% Status : Computes Auxiliary variables of the dynamic model and returns a dseries
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

ds.AUX_EXO_LAG_13_0=ds.eps_a_news;
ds.AUX_EXO_LAG_13_1=ds.AUX_EXO_LAG_13_0(-1);
ds.AUX_EXO_LAG_13_2=ds.AUX_EXO_LAG_13_1(-1);
ds.AUX_EXO_LAG_13_3=ds.AUX_EXO_LAG_13_2(-1);
ds.AUX_EXO_LAG_13_4=ds.AUX_EXO_LAG_13_3(-1);
ds.AUX_EXO_LAG_13_5=ds.AUX_EXO_LAG_13_4(-1);
ds.AUX_EXO_LAG_13_6=ds.AUX_EXO_LAG_13_5(-1);
ds.AUX_EXO_LAG_13_7=ds.AUX_EXO_LAG_13_6(-1);
end
