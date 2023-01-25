function SimuRslt = simulate_rbcinv(IterRslt,GNDSGE_OPTIONS)
%% Add path
if ispc
    BLAS_FILE = 'essential_blas.dll';
    PATH_DELIMITER = ';';
    
    GDSGE_TOOLBOX_ROOT = fileparts(which(BLAS_FILE));
    if ~any(strcmp(strsplit(getenv('PATH'),PATH_DELIMITER),GDSGE_TOOLBOX_ROOT))
        setenv('PATH',[getenv('PATH'),PATH_DELIMITER,GDSGE_TOOLBOX_ROOT]);
    end
    
    clear BLAS_FILE PATH_DELIMITER GDSGE_TOOLBOX_ROOT
elseif ismac
    if exist('./essential_blas.dylib','file') == 0
        copyfile(which('essential_blas.dylib'),'./');
    end
end

%% Simulate code starts here
TolEq = 1e-6;
TolSol = 1e-8;
TolFun = 1e-8;
PrintFreq = 10;
NoPrint = 0;
SaveFreq = 10;
NoSave = 0;
SimuPrintFreq = 1000;
SimuSaveFreq = inf;
NumThreads = feature('numcores');
MaxIter = inf;
MaxMinorIter = inf;
num_samples = 1;
num_periods = 1000;
SolMaxIter = 200;

% task constants
MEX_TASK_INIT = 0;
MEX_TASK_INF_HORIZON = 1;

% Solver
UseFiniteDiff = 0;
UseBroyden = 0;
FiniteDiffDelta = 1e-6;

% DEBUG flag
GNDSGE_DEBUG_EVAL_ONLY = 0;
GNDSGE_USE_BROYDEN = 1;
GNDSGE_USE_BROYDEN_NOW = 0;

v2struct(IterRslt.pp);

INTERP_ORDER = 4;
EXTRAP_ORDER = 2;
OutputInterpOrder = 2;
USE_SPLINE = 1;
GNDSGE_USE_OLD_VEC = 0;
USE_ASG = 0;
USE_PCHIP = 0;
SIMU_INTERP = 0;
SIMU_RESOLVE = 1;
SimuSeed = 0823;
AsgMinLevel = 4;
AsgMaxLevel = 10;
AsgThreshold = 1e-2;
AsgOutputMaxLevel = 10;
AsgOutputThreshold = 1e-2;
IterSaveAll = 0;
SkipModelInit = 0;
GNDSGE_EMPTY = [];
GNDSGE_ASG_FIX_GRID = 0;
UseModelId = 0;
MinBatchSol = 1;
UseAdaptiveBound = 1;
UseAdaptiveBoundInSol = 0;
EnforceSimuStateInbound = 1;
REMOVE_NULL_STATEMENTS = 0;
REUSE_WARMUP_SOL = 1;
INTERP_WARMUP_SOL = 1;
CONSTRUCT_OUTPUT = 1;
shock_num = 1;
shock_trans = 1;
GNDSGE_dummy_state = 1;
GNDSGE_dummy_shock = 1;
DEFAULT_PARAMETERS_END_HERE = true;
beta  = 0.98;
sigma = 2.0;
alpha = 0.36;
delta = 0.025;
phi   = 0.975;
shock_num = 9;
z = [-0.009000000000000001, -0.006750000000000001, -0.0045000000000000005, -0.0022500000000000003, 0.0, 0.0022500000000000003, 0.0045000000000000005, 0.006750000000000001, 0.009000000000000001];
shock_trans = [0.2997915954686959 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.518962806592363e-5;
0.1468590563758959 0.23522952143515136 0.29155620190103265 0.21128555006621186 0.08948161070006966 0.022121085718597944 0.0031866805262243947 0.00026694752780032083 0.0002802932768162236;
0.05762822227615314 0.14705757307109943 0.2654215605998526 0.28005476158111775 0.17275928191970524 0.062263542359461965 0.013092777076742279 0.0016033564085408125 0.0017222811158675855;
0.017864420562816546 0.07064357087458546 0.18574512631267154 0.28536457462016895 0.25632218228299797 0.13458865731311143 0.04127393210905195 0.007381183611767539 0.008197535924596155;
0.004332448363012557 0.026063913402248794 0.09989815537154749 0.22353571619046736 0.29233953334544754 0.22353571619046742 0.09989815537154745 0.026063913402248784 0.030396361765261393;
0.000816352312828562 0.007381183611767554 0.04127393210905196 0.1345886573131114 0.256322182282998 0.28536457462016895 0.18574512631267148 0.0706435708745855 0.08850799143740207;
0.00011892470732676076 0.0016033564085408615 0.013092777076742284 0.062263542359461965 0.17275928191970513 0.2800547615811178 0.26542156059985267 0.14705757307109935 0.2046857953472525;
1.3345749015906309e-5 0.0002669475278002703 0.003186680526224491 0.022121085718597944 0.08948161070006966 0.2112855500662118 0.2915562019010327 0.2352295214351513 0.3820885778110472;
1.1505767815436636e-6 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5890103628687295];
shock_num = 9;
v = [-0.026000000000000002, -0.0245, -0.023000000000000003, -0.021500000000000002, -0.02, -0.018500000000000003, -0.017, -0.0155, -0.014];
shock_trans = [0.2997915954686959 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.518962806592363e-5;
0.1468590563758959 0.23522952143515136 0.29155620190103265 0.21128555006621186 0.08948161070006966 0.022121085718597944 0.0031866805262243947 0.00026694752780032083 0.0002802932768162236;
0.05762822227615314 0.14705757307109943 0.2654215605998526 0.28005476158111775 0.17275928191970524 0.062263542359461965 0.013092777076742279 0.0016033564085408125 0.0017222811158675855;
0.017864420562816546 0.07064357087458546 0.18574512631267154 0.28536457462016895 0.25632218228299797 0.13458865731311143 0.04127393210905195 0.007381183611767539 0.008197535924596155;
0.004332448363012557 0.026063913402248794 0.09989815537154749 0.22353571619046736 0.29233953334544754 0.22353571619046742 0.09989815537154745 0.026063913402248784 0.030396361765261393;
0.000816352312828562 0.007381183611767554 0.04127393210905196 0.1345886573131114 0.256322182282998 0.28536457462016895 0.18574512631267148 0.0706435708745855 0.08850799143740207;
0.00011892470732676076 0.0016033564085408615 0.013092777076742284 0.062263542359461965 0.17275928191970513 0.2800547615811178 0.26542156059985267 0.14705757307109935 0.2046857953472525;
1.3345749015906309e-5 0.0002669475278002703 0.003186680526224491 0.022121085718597944 0.08948161070006966 0.2112855500662118 0.2915562019010327 0.2352295214351513 0.3820885778110472;
1.1505767815436636e-6 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5890103628687295];
Kss  = (alpha/(1/beta - 1 + delta))^(1/(1-alpha));
Iss = Kss*delta;
KPts = 101;
KMin = Kss*0.5;
KMax = Kss*1.5;
K    = linspace(KMin,KMax,KPts);
num_periods = 10000;
num_samples = 100;


GEN_SHOCK_START_PERIOD = 1;

shock_trans = IterRslt.shock_trans;
v2struct(IterRslt.params);
v2struct(IterRslt.var_shock);
v2struct(IterRslt.var_state);

if nargin>=2
    v2struct(GNDSGE_OPTIONS)
end

%% Construct interpolation for solutions
if shock_num>1
    GNDSGE_PP=struct('form','MKL','breaks',{{1:shock_num,K}},...
        'Values',reshape(IterRslt.GNDSGE_PROB.GNDSGE_SOL, [numel(IterRslt.GNDSGE_PROB.GNDSGE_SOL)/prod(IterRslt.GNDSGE_PROB.GNDSGE_SIZE),IterRslt.GNDSGE_PROB.GNDSGE_SIZE]),...
        'coefs',[],'order',[2 OutputInterpOrder*ones(1,length({K}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
else
    GNDSGE_PP=struct('form','MKL','breaks',{{K}},...
        'Values',reshape(IterRslt.GNDSGE_PROB.GNDSGE_SOL, [numel(IterRslt.GNDSGE_PROB.GNDSGE_SOL)/prod(IterRslt.GNDSGE_PROB.GNDSGE_SIZE),IterRslt.GNDSGE_PROB.GNDSGE_SIZE(2:end)]),...
        'coefs',[],'order',[OutputInterpOrder*ones(1,length({K}))],'Method',[],...
        'ExtrapolationOrder',[],'thread',NumThreads, ...
        'orient','curvefit');
end
GNDSGE_PP=myinterp(GNDSGE_PP);

GNDSGE_NPROB = num_samples;

SimuRslt.shock = ones(num_samples,num_periods+1);
SimuRslt.K=zeros(num_samples,num_periods);
SimuRslt.c=zeros(num_samples,num_periods);
SimuRslt.K=zeros(num_samples,num_periods);
SimuRslt.w=zeros(num_samples,num_periods);
SimuRslt.Inv=zeros(num_samples,num_periods);


SimuRslt.K(:,1)=Kss;
SimuRslt.shock(:,1)=1;


if nargin>1 && isfield(GNDSGE_OPTIONS,'init')
    SimuRslt.K(:,1:size(GNDSGE_OPTIONS.init.K,2))=GNDSGE_OPTIONS.init.K;
SimuRslt.shock(:,1:size(GNDSGE_OPTIONS.init.shock,2))=GNDSGE_OPTIONS.init.shock;

end

if any(SimuRslt.shock(:,1)>shock_num)
    error('initial shock exceeds shock_num');
end

GNDSGE_LB = zeros(3,GNDSGE_NPROB);
GNDSGE_UB = 100*ones(3,GNDSGE_NPROB);
GNDSGE_LB(1:1,:)=0;
GNDSGE_LB(3:3,:)=0;
GNDSGE_UB(3:3,:)=1.0;



% Use the largest bound in IterSol
if UseAdaptiveBound==1
GNDSGE_LB = repmat(min(IterRslt.GNDSGE_PROB.GNDSGE_LB,[],2),[1,GNDSGE_NPROB]);
GNDSGE_UB = repmat(max(IterRslt.GNDSGE_PROB.GNDSGE_UB,[],2),[1,GNDSGE_NPROB]);
end

GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
GNDSGE_EQVAL = 1e20*ones(3,GNDSGE_NPROB);
GNDSGE_F = 1e20*ones(1,GNDSGE_NPROB);
GNDSGE_SOL = zeros(3,GNDSGE_NPROB);
GNDSGE_AUX = zeros(2,GNDSGE_NPROB);

%% Generate random number
SimuRslt.shock(:,GEN_SHOCK_START_PERIOD:end) = gen_discrete_markov_rn(shock_trans,num_samples,length(GEN_SHOCK_START_PERIOD:num_periods+1),...
    SimuRslt.shock(:,GEN_SHOCK_START_PERIOD));

MEX_TASK_NAME = MEX_TASK_INF_HORIZON;
GNDSGE_SHOCK_VAR_INDEX_BASE = ([0:num_samples-1]')*shock_num;

tic;
for GNDSGE_t=1:num_periods
    % Reuse the init inbound and tensor code
    % Map grid to current variable
    shock = SimuRslt.shock(:,GNDSGE_t);
    K=SimuRslt.K(:,GNDSGE_t);

    
    GNDSGE_data0 = repmat([shock_num;beta(:);sigma(:);alpha(:);delta(:);phi(:);Iss(:);shock_trans(:);z(:);v(:); ],1,GNDSGE_NPROB);
    
    
    
    % Use interpolation as initial values
%     %{
    if GNDSGE_t>0
        if shock_num>1
            GNDSGE_SOL = myinterp_mex(int32(NumThreads),GNDSGE_PP.breaks,GNDSGE_PP.coefs,...
                int32(GNDSGE_PP.pieces),int32(GNDSGE_PP.order),int32(GNDSGE_PP.dim),'not-a-knot',[SimuRslt.shock(:,GNDSGE_t)';SimuRslt.K(:,GNDSGE_t)'],[],[],[]);
        else
            GNDSGE_SOL = myinterp_mex(int32(NumThreads),GNDSGE_PP.breaks,GNDSGE_PP.coefs,...
                int32(GNDSGE_PP.pieces),int32(GNDSGE_PP.order),int32(GNDSGE_PP.dim),'not-a-knot',[SimuRslt.K(:,GNDSGE_t)'],[],[],[]);
        end
    end
    %}
    
    if UseAdaptiveBoundInSol==1
        % Tentatively adjust the bound
        GNDSGE_LB_OLD = GNDSGE_LB;
        GNDSGE_UB_OLD = GNDSGE_UB;
        
        
        
        % Hitting lower bound
        GNDSGE_SOL_hitting_lower_bound = abs(GNDSGE_SOL - GNDSGE_LB_OLD) < 1e-8;
        GNDSGE_SOL_hitting_upper_bound = abs(GNDSGE_SOL - GNDSGE_UB_OLD) < 1e-8;
        
        % Adjust for those hitting lower bound or upper bound
        GNDSGE_LB(~GNDSGE_SOL_hitting_lower_bound) = GNDSGE_LB_OLD(~GNDSGE_SOL_hitting_lower_bound);
        GNDSGE_UB(~GNDSGE_SOL_hitting_upper_bound) = GNDSGE_UB_OLD(~GNDSGE_SOL_hitting_upper_bound);
    end

    % Construct tensor variable
    
    
    % Reconstruct data
    GNDSGE_DATA = [GNDSGE_data0;shock(:)';K(:)';   ];
    
    % Solve problems
    GNDSGE_SKIP = zeros(1,GNDSGE_NPROB);
    [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_rbcinv(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    while (max(isnan(GNDSGE_F)) || max(GNDSGE_F(:))>TolSol)
        GNDSGE_X0Rand = rand(size(GNDSGE_SOL)) .* (GNDSGE_UB-GNDSGE_LB) + GNDSGE_LB;
        NeedResolved = (GNDSGE_F>TolSol) | isnan(GNDSGE_F);
        GNDSGE_SOL(:,NeedResolved) = GNDSGE_X0Rand(:,NeedResolved);
        GNDSGE_SKIP(~NeedResolved) = 1;
        
        [GNDSGE_SOL,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL,GNDSGE_OPT_INFO] = mex_rbcinv(GNDSGE_SOL,GNDSGE_LB,GNDSGE_UB,GNDSGE_DATA,GNDSGE_SKIP,GNDSGE_F,GNDSGE_AUX,GNDSGE_EQVAL);
    
        if UseAdaptiveBoundInSol==1
            % Tentatively adjust the bound
            GNDSGE_LB_OLD = GNDSGE_LB;
            GNDSGE_UB_OLD = GNDSGE_UB;
            
            
            
            % Hitting lower bound
            GNDSGE_SOL_hitting_lower_bound = abs(GNDSGE_SOL - GNDSGE_LB_OLD) < 1e-8;
            GNDSGE_SOL_hitting_upper_bound = abs(GNDSGE_SOL - GNDSGE_UB_OLD) < 1e-8;
            
            % Adjust for those hitting lower bound or upper bound
            GNDSGE_LB(~GNDSGE_SOL_hitting_lower_bound) = GNDSGE_LB_OLD(~GNDSGE_SOL_hitting_lower_bound);
            GNDSGE_UB(~GNDSGE_SOL_hitting_upper_bound) = GNDSGE_UB_OLD(~GNDSGE_SOL_hitting_upper_bound);
        end
    end
    
    c=GNDSGE_SOL(1:1,:);
K_next=GNDSGE_SOL(2:2,:);
mu=GNDSGE_SOL(3:3,:);

    
    w=GNDSGE_AUX(1,:);
Inv=GNDSGE_AUX(2,:);

    
    GNDSGE_SHOCK_VAR_LINEAR_INDEX = SimuRslt.shock(:,GNDSGE_t+1) + GNDSGE_SHOCK_VAR_INDEX_BASE;
    SimuRslt.c(:,GNDSGE_t)=c;
SimuRslt.K(:,GNDSGE_t)=K;
SimuRslt.w(:,GNDSGE_t)=w;
SimuRslt.Inv(:,GNDSGE_t)=Inv;
SimuRslt.K(:,GNDSGE_t+1) = K_next(:);



    
    
    
    if mod(GNDSGE_t,SimuPrintFreq)==0
        fprintf('Periods: %d\n', GNDSGE_t);
        SimuRsltNames = fieldnames(SimuRslt);
        for GNDSGE_field = 1:length(SimuRsltNames)
            fprintf('%8s', SimuRsltNames{GNDSGE_field});
        end
        fprintf('\n');
        for GNDSGE_field = 1:length(SimuRsltNames)
            fprintf('%8.4g', SimuRslt.(SimuRsltNames{GNDSGE_field})(1,GNDSGE_t));
        end
        fprintf('\n');
        toc;
        tic;
    end
    
    if mod(GNDSGE_t,SimuSaveFreq)==0
        save(['SimuRslt_rbcinv_' num2str(GNDSGE_t) '.mat'], 'SimuRslt');
    end
end
end