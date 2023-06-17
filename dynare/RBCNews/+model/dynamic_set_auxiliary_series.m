function ds = dynamic_set_auxiliary_series(ds, params)
%
% Status : Computes Auxiliary variables of the dynamic model and returns a dseries
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

ds.AUX_EXO_LAG_8_0=ds.eps_z_news;
ds.AUX_EXO_LAG_8_1=ds.AUX_EXO_LAG_8_0(-1);
ds.AUX_EXO_LAG_8_2=ds.AUX_EXO_LAG_8_1(-1);
ds.AUX_EXO_LAG_8_3=ds.AUX_EXO_LAG_8_2(-1);
ds.AUX_EXO_LAG_8_4=ds.AUX_EXO_LAG_8_3(-1);
ds.AUX_EXO_LAG_8_5=ds.AUX_EXO_LAG_8_4(-1);
ds.AUX_EXO_LAG_8_6=ds.AUX_EXO_LAG_8_5(-1);
ds.AUX_EXO_LAG_8_7=ds.AUX_EXO_LAG_8_6(-1);
end
