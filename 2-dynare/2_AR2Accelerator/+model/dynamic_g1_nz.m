function [nzij_pred, nzij_current, nzij_fwrd] = dynamic_g1_nz()
% Returns the coordinates of non-zero elements in the Jacobian, in column-major order, for each lead/lag (only for endogenous)
  nzij_pred = zeros(3, 2, 'int32');
  nzij_pred(1,1)=3; nzij_pred(1,2)=1;
  nzij_pred(2,1)=1; nzij_pred(2,2)=2;
  nzij_pred(3,1)=2; nzij_pred(3,2)=4;
  nzij_current = zeros(8, 2, 'int32');
  nzij_current(1,1)=1; nzij_current(1,2)=1;
  nzij_current(2,1)=3; nzij_current(2,2)=1;
  nzij_current(3,1)=4; nzij_current(3,2)=1;
  nzij_current(4,1)=4; nzij_current(4,2)=2;
  nzij_current(5,1)=3; nzij_current(5,2)=3;
  nzij_current(6,1)=4; nzij_current(6,2)=3;
  nzij_current(7,1)=2; nzij_current(7,2)=4;
  nzij_current(8,1)=4; nzij_current(8,2)=4;
  nzij_fwrd = zeros(0, 2, 'int32');
end
