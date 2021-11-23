This archive contains three folders with Matlab, Python, and Julia code for solving discrete-time income fluctuation problems. The original Matlab code was written by Greg Kaplan. The Python and Julia translations were written by Tom Sweeney.

The folder "Matlab" contains Matlab code to solve the following discrete time models:
 - vfi_deterministic.m: deterministic infinite horizon consumption-savings problem using value function iteration
 - vfi_deterministic_finite.m: deterministic finite horizon consumption-savings problem using value function iteration
 - vfi_IID.m: consumption-savings problem with IID income using value function iteration
 - eei_IID.m: consumption-savings problem with IID income using euler equation iteration, uses auxillary function fn_eeqn_c.m
 - egp_IID.m: consumption-savings problem with IID income using endogenous gridpoints
 - egp_AR1_IID.m: consumption-savings problem with income process consisting of AR(1) + IID in logs, using endogenous gridpoints with cash on hand as a state variable
 - egp_IID_lifecycle.m: lifecycle consumption-savings problem with retirement period with IID income using endogenous gridpoints
 - egp_IID_aiyagari.m: general equilibrium consumption-savings problem as in Aiyagari (94) with IID income using endogenous gridpoints
 
The following are auxillary routines that are used in the above programs:
 - lininterp1.m: a fast, simple linear interpolation routine
 - discrete_normal.m: discretizes a Normal distribution
 - rouwenhorst.m: discretizes an AR(1) process using the Rouwenhorst method
 
The folders "Python" and "Julia" contain analogous Python and Julia code.