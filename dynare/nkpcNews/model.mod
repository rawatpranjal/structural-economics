/*
 * Copyright (C) 2016-2021 Johannes Pfeifer
 * Copyright (C) 2021 Willi Mutschler
 *
 * This is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * For a copy of the GNU General Public License,
 * see <http://www.gnu.org/licenses/>.
 */



var
  a           ${a}$                   (long_name='technology shock process (log dev ss)')
  z           ${z}$                   (long_name='preference shock process (log dev ss)')
  c           ${c}$                   (long_name='consumption (log dev ss)')
  y           ${y}$                   (long_name='output (log dev ss)')
  y_nat       ${y^{nat}}$             (long_name='natural output (log dev ss)')
  y_gap       ${\tilde y}$            (long_name='output gap (log dev ss)')
  r_nat       ${r^{nat}}$             (long_name='natural interest rate (log dev ss)')
  r_real      ${r}$                   (long_name='real interest rate (log dev ss)')     
  ii          ${i}$                   (long_name='nominal interest rate (log dev ss)')  
  pie         ${\pi}$                 (long_name='inflation (log dev ss)')
  n           ${n}$                   (long_name='hours worked (log dev ss)')
  w           ${w}$                   (long_name='real wage (log dev ss)')
;     


varexo  
  eps_a       ${\varepsilon_a}$       (long_name='technology shock')
  eps_z       ${\varepsilon_z}$       (long_name='preference shock')
;

parameters 
  ALPHA      ${\alpha}$              (long_name='one minus labor share in production')
  BETA       ${\beta}$               (long_name='discount factor')
  RHOA       ${\rho_a}$              (long_name='autocorrelation technology process')
  RHOZ       ${\rho_{z}}$            (long_name='autocorrelation preference process')
  SIGMA      ${\sigma}$              (long_name='inverse EIS')
  VARPHI     ${\varphi}$             (long_name='inverse Frisch elasticity')
  EPSILON    ${\epsilon}$            (long_name='Dixit-Stiglitz demand elasticity')
  THETA      ${\theta}$              (long_name='Calvo probability')
  @#if MONPOL != 1
  PHI_PIE    ${\phi_{\pi}}$          (long_name='inflation feedback Taylor Rule')
  PHI_Y      ${\phi_{y}}$            (long_name='output feedback Taylor Rule')
  @#endif
;

model(linear); 
//Composite parameters
#OMEGA=(1-ALPHA)/(1-ALPHA+ALPHA*EPSILON);
#PSI_YA=(1+VARPHI)/(SIGMA*(1-ALPHA)+VARPHI+ALPHA);
#LAMBDA=(1-THETA)*(1-BETA*THETA)/THETA*OMEGA;
#KAPPA=LAMBDA*(SIGMA+(VARPHI+ALPHA)/(1-ALPHA));

[name='New Keynesian Phillips Curve']
pie=BETA*pie(+1)+KAPPA*y_gap;

[name='Dynamic IS Curve']
y_gap=-1/SIGMA*(ii-pie(+1)-r_nat)+y_gap(+1);

[name='Production function']
y=a+(1-ALPHA)*n;

[name='labor demand']
w = SIGMA*c+VARPHI*n;

[name='resource constraint']
y=c;

[name='TFP process']
a=RHOA*a(-1)+eps_a;

[name='Preference shifter']
z = RHOZ*z(-1) + eps_z;

[name='Definition natural rate of interest']
r_nat=-SIGMA*PSI_YA*(1-RHOA)*a+(1-RHOZ)*z;

[name='Definition real interest rate']
r_real=ii-pie(+1);

[name='Definition natural output']
y_nat=PSI_YA*a;

[name='Definition output gap']
y_gap=y-y_nat;

@#if MONPOL == 1
[name='Interest Rate Rule: Exogenous One-To-One']
ii = r_nat;
@#elseif MONPOL == 2
ii = r_nat + PHI_PIE*pie+PHI_Y*y_gap;
@#else
ii = PHI_PIE*pie+PHI_Y*y;
@#endif

end;

shocks;
    var eps_z  = 1;
    var eps_a  = 1;
end;

