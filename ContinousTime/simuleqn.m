syms s1 s2 s3 s4;
eqn1 = s1+s2+s3+s4 == 1;
eqn2 = s2+s3==1/4;
eqn3 = s1+s4==3/4;
eqn4 = s3==1/9;
eqn5 = s4==1/3;
eqn6 = s1+s2==5/9;

sol = solve([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6], [s1, s2, s3, s4]);
sol.s1
sol.s2
sol.s3
sol.s4