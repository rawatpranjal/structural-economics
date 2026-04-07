function [nzij_pred, nzij_current, nzij_fwrd] = dynamic_g1_nz()
% Returns the coordinates of non-zero elements in the Jacobian, in column-major order, for each lead/lag (only for endogenous)
  nzij_pred = zeros(0, 2, 'int32');
  nzij_current = zeros(1, 2, 'int32');
  nzij_current(1,1)=1; nzij_current(1,2)=1;
  nzij_fwrd = zeros(1, 2, 'int32');
  nzij_fwrd(1,1)=1; nzij_fwrd(1,2)=1;
end
