function f = fn_eeqn_c(a)

global agrid conlast u1 beta ny R ydist cash;

c = zeros(1,ny);
for iy = 1:ny
    c(iy) = lininterp1(agrid,conlast(:,iy),a);
end

f = u1(cash-a)-beta.*R.*(u1(c)*ydist);



