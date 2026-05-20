# Catalog Audit Index

`bullshit-detector` audit of all 97 tutorials. Each row links to the
skill-native audit file inside that tutorial's folder. Claim source =
`README.md`; code = `run.py`; data = `tables/*.csv`.

**Status:** Stage 1 complete (97/97 audited). Stage 2 complete (19/19
highest-risk tutorials fixed and re-audited <=25%). Stage 3 complete (all
70 remaining tutorials fixed by opus and re-checked by sonnet; every
tutorial now re-audits to 0-15%, all <=25%; 3 needed a second pass).
Stage 4 (README/code split) pending.

## Stage 3 fix results

Opus subagents fixed the 70 mop-up tutorials (TDD red/green tests),
sonnet check subagents re-audited each. Outcome: 67 CONFIRMED on the
first check, 3 required a residual second pass (`theory-of-the-firm`,
`hamiltonian-monte-carlo`, `dsge/behavioral-nk`). After fixes, every
Stage-3 tutorial re-audits at 0-15% (most at 0%); the residual non-zero
scores are LOW-severity DATA DRIFT (runtime scalars not yet in a CSV) or
prose nuance, all within the <=25% exit target. Per-tutorial recheck
audits are in each folder as `bullshit-detector_<stem>_recheck_2026-05-20.md`.

## Stage 1 summary

- 97 tutorials audited, one `bullshit-detector` subagent each.
- 8 came back HOLDS (no faithfulness gap): `logit-discrete-choice`,
  `maximum-score-binary-choice`, `money-pump-index`,
  `revealed-price-preference`, `rum-choice-networks`,
  `quantal-response-equilibrium`, `zero-intelligence-traders`, `nkdsge`.
- 18 carry a FALSE finding (code or prose does the opposite of a claim).
- Highest score: `numerical-methods/global-search-multistart` at 75% (the
  single-start baseline finds the global optimum, inverting the tutorial's
  whole lesson). No tutorial scored above 75%.
- Most findings are DATA DRIFT (runtime scalars in README prose with no
  backing CSV) or DILUTED (prose slightly overstates what code does).

### Stage 2 shortlist (highest-risk: score >=50% OR a FALSE finding)

19 tutorials. Fixed by opus subagents, substance before form:

`numerical-methods/global-search-multistart` (75),
`choice/consideration-set-estimation` (65),
`choice/nested-logit` (65),
`choice/probability-distortion-mixture` (65),
`industrial-organization/dynamic-entry-exit` (60),
`heterogeneous-agents/huggett-aggregate-risk-srl` (55),
`numerical-methods/interpolation` (50),
`heterogeneous-agents/aiyagari-hact` (40),
`choice/convex-time-budget-present-bias` (35),
`dynamic-programming/cake-eating` (35),
`structural-econometrics/keane-wolpin-career-choice` (35),
`numerical-methods/fixed-point-acceleration` (35),
`bayesian-methods/neural-posterior-brock-hommes` (35),
`numerical-methods/bayesian-optimization` (30),
`structural-econometrics/adversarial-estimation` (25),
`numerical-methods/constrained-optimization-kkt` (25),
`numerical-methods/root-finding` (25),
`heterogeneous-agents/endogenous-grid-points` (25),
`optimal-control/phase-diagrams` (25).

The remaining ~70 tutorials with MISLABELED / DATA DRIFT / MED-LOW DILUTED
findings are cleared in Stage 3.

## Stage 2 fix results

Opus subagent fixes (TDD red/green tests), independently re-audited by a
sonnet check subagent. CONFIRMED = re-audit <=25% and honest-fix tests pass.

| Tutorial | Old | New | Verdict |
|----------|-----|-----|---------|
| numerical-methods/global-search-multistart | 75% | 10% | CONFIRMED |
| choice/consideration-set-estimation | 65% | 15% | CONFIRMED |
| choice/nested-logit | 65% | 10% | CONFIRMED |
| choice/probability-distortion-mixture | 65% | 10% | CONFIRMED |
| industrial-organization/dynamic-entry-exit | 60% | 5% | CONFIRMED |
| heterogeneous-agents/huggett-aggregate-risk-srl | 55% | 15% | CONFIRMED |
| numerical-methods/interpolation | 50% | 0% | CONFIRMED |
| heterogeneous-agents/aiyagari-hact | 40% | 15% | CONFIRMED |
| choice/convex-time-budget-present-bias | 35% | 20% | CONFIRMED |
| dynamic-programming/cake-eating | 35% | 10% | CONFIRMED |
| structural-econometrics/keane-wolpin-career-choice | 35% | 5% | CONFIRMED |
| numerical-methods/fixed-point-acceleration | 35% | 0% | CONFIRMED |
| bayesian-methods/neural-posterior-brock-hommes | 35% | 5% | CONFIRMED |
| numerical-methods/bayesian-optimization | 30% | 10% | CONFIRMED |
| structural-econometrics/adversarial-estimation | 25% | 5% | CONFIRMED |
| numerical-methods/constrained-optimization-kkt | 25% | 0% | CONFIRMED |
| numerical-methods/root-finding | 25% | 5% | CONFIRMED |
| heterogeneous-agents/endogenous-grid-points | 25% | 0% | CONFIRMED |
| optimal-control/phase-diagrams | 25% | 0% | CONFIRMED |

## Scores

| # | Tutorial | Score | Worst category | Result-changing? | Audit file |
|---|----------|-------|----------------|------------------|------------|
| 1 | choice/bayesian-learning | 25% | DILUTED | yes | [file](choice/bayesian-learning/bullshit-detector_bayesian-learning_2026-05-20.md) |
| 2 | choice/consideration-set-estimation | 65% | FALSE | yes | [file](choice/consideration-set-estimation/bullshit-detector_consideration-set-estimation_2026-05-20.md) |
| 3 | choice/convex-time-budget-present-bias | 35% | FALSE | no | [file](choice/convex-time-budget-present-bias/bullshit-detector_convex-time-budget-present-bias_2026-05-20.md) |
| 4 | choice/houtman-maks-rational-subsets | 10% | DILUTED | no | [file](choice/houtman-maks-rational-subsets/bullshit-detector_houtman-maks-rational-subsets_2026-05-20.md) |
| 5 | choice/logit-discrete-choice | 5% | HOLDS | no | [file](choice/logit-discrete-choice/bullshit-detector_logit-discrete-choice_2026-05-20.md) |
| 6 | choice/maximum-score-binary-choice | 0% | HOLDS | no | [file](choice/maximum-score-binary-choice/bullshit-detector_maximum-score-binary-choice_2026-05-20.md) |
| 7 | choice/mixed-logit-simulation | 10% | DILUTED | no | [file](choice/mixed-logit-simulation/bullshit-detector_mixed-logit-simulation_2026-05-20.md) |
| 8 | choice/money-pump-index | 0% | HOLDS | no | [file](choice/money-pump-index/bullshit-detector_money-pump-index_2026-05-20.md) |
| 9 | choice/nested-logit | 65% | FALSE | yes | [file](choice/nested-logit/bullshit-detector_nested-logit_2026-05-20.md) |
| 10 | choice/preference-recoverability | 20% | DILUTED | no | [file](choice/preference-recoverability/bullshit-detector_preference-recoverability_2026-05-20.md) |
| 11 | choice/probability-distortion-mixture | 65% | FALSE | yes | [file](choice/probability-distortion-mixture/bullshit-detector_probability-distortion-mixture_2026-05-20.md) |
| 12 | choice/revealed-preference-afriat | 20% | MISLABELED | no | [file](choice/revealed-preference-afriat/bullshit-detector_revealed-preference-afriat_2026-05-20.md) |
| 13 | choice/revealed-price-preference | 5% | HOLDS | no | [file](choice/revealed-price-preference/bullshit-detector_revealed-price-preference_2026-05-20.md) |
| 14 | choice/risk-aversion-monotone-choice | 10% | DATA DRIFT | no | [file](choice/risk-aversion-monotone-choice/bullshit-detector_risk-aversion-monotone-choice_2026-05-20.md) |
| 15 | choice/sequential-search-ursu | 35% | DILUTED | yes | [file](choice/sequential-search-ursu/bullshit-detector_sequential-search-ursu_2026-05-20.md) |
| 16 | choice/urn-behavioral-mixtures | 15% | DATA DRIFT | no | [file](choice/urn-behavioral-mixtures/bullshit-detector_urn-behavioral-mixtures_2026-05-20.md) |
| 17 | industrial-organization/blp-random-coefficients | 20% | DATA DRIFT | no | [file](industrial-organization/blp-random-coefficients/bullshit-detector_blp-random-coefficients_2026-05-20.md) |
| 18 | industrial-organization/dynamic-discrete-choice | 25% | DILUTED | no | [file](industrial-organization/dynamic-discrete-choice/bullshit-detector_dynamic-discrete-choice_2026-05-20.md) |
| 19 | industrial-organization/dynamic-entry-exit | 60% | DILUTED | yes | [file](industrial-organization/dynamic-entry-exit/bullshit-detector_dynamic-entry-exit_2026-05-20.md) |
| 20 | industrial-organization/dynamic-games-estimation | 25% | DILUTED | no | [file](industrial-organization/dynamic-games-estimation/bullshit-detector_dynamic-games-estimation_2026-05-20.md) |
| 21 | industrial-organization/dynamic-games | 35% | DILUTED | yes | [file](industrial-organization/dynamic-games/bullshit-detector_dynamic-games_2026-05-20.md) |
| 22 | industrial-organization/logit-supply-side | 20% | MISLABELED | no | [file](industrial-organization/logit-supply-side/bullshit-detector_logit-supply-side_2026-05-20.md) |
| 23 | industrial-organization/merger-simulation | 40% | DILUTED | yes | [file](industrial-organization/merger-simulation/bullshit-detector_merger-simulation_2026-05-20.md) |
| 24 | industrial-organization/nash-in-nash | 10% | DILUTED | no | [file](industrial-organization/nash-in-nash/bullshit-detector_nash-in-nash_2026-05-20.md) |
| 25 | industrial-organization/production-functions-markups | 40% | DILUTED | no | [file](industrial-organization/production-functions-markups/bullshit-detector_production-functions-markups_2026-05-20.md) |
| 26 | industrial-organization/theory-of-the-firm | 15% | DILUTED | no | [file](industrial-organization/theory-of-the-firm/bullshit-detector_theory-of-the-firm_2026-05-20.md) |
| 27 | industrial-organization/vertical-contracts | 25% | DILUTED | no | [file](industrial-organization/vertical-contracts/bullshit-detector_vertical-contracts_2026-05-20.md) |
| 28 | industrial-organization/vertical-relationships | 20% | MISLABELED | no | [file](industrial-organization/vertical-relationships/bullshit-detector_vertical-relationships_2026-05-20.md) |
| 29 | dynamic-programming/aiyagari | 15% | DATA DRIFT | no | [file](dynamic-programming/aiyagari/bullshit-detector_aiyagari_2026-05-20.md) |
| 30 | dynamic-programming/asset-pricing | 10% | DATA DRIFT | no | [file](dynamic-programming/asset-pricing/bullshit-detector_asset-pricing_2026-05-20.md) |
| 31 | dynamic-programming/cake-eating | 35% | FALSE | no | [file](dynamic-programming/cake-eating/bullshit-detector_cake-eating_2026-05-20.md) |
| 32 | dynamic-programming/consumption-savings | 20% | MISLABELED | no | [file](dynamic-programming/consumption-savings/bullshit-detector_consumption-savings_2026-05-20.md) |
| 33 | dynamic-programming/diamond-mortensen-pissarides | 25% | DILUTED | no | [file](dynamic-programming/diamond-mortensen-pissarides/bullshit-detector_diamond-mortensen-pissarides_2026-05-20.md) |
| 34 | dynamic-programming/job-search-mccall | 5% | DATA DRIFT | no | [file](dynamic-programming/job-search-mccall/bullshit-detector_job-search-mccall_2026-05-20.md) |
| 35 | dynamic-programming/optimal-growth | 20% | DILUTED | no | [file](dynamic-programming/optimal-growth/bullshit-detector_optimal-growth_2026-05-20.md) |
| 36 | dynamic-programming/q-learning-growth | 25% | DILUTED | no | [file](dynamic-programming/q-learning-growth/bullshit-detector_q-learning-growth_2026-05-20.md) |
| 37 | dynamic-programming/rbc | 10% | DATA DRIFT | no | [file](dynamic-programming/rbc/bullshit-detector_rbc_2026-05-20.md) |
| 38 | dynamic-programming/shock-discretization | 10% | DILUTED | no | [file](dynamic-programming/shock-discretization/bullshit-detector_shock-discretization_2026-05-20.md) |
| 39 | dynamic-programming/solow-growth | 20% | DILUTED | no | [file](dynamic-programming/solow-growth/bullshit-detector_solow-growth_2026-05-20.md) |
| 40 | computational-methods/hamiltonian-monte-carlo | 40% | DILUTED | no | [file](computational-methods/hamiltonian-monte-carlo/bullshit-detector_hamiltonian-monte-carlo_2026-05-20.md) |
| 41 | computational-methods/kalman-filter | 15% | DILUTED | no | [file](computational-methods/kalman-filter/bullshit-detector_kalman-filter_2026-05-20.md) |
| 42 | computational-methods/metropolis-hastings | 15% | DATA DRIFT | no | [file](computational-methods/metropolis-hastings/bullshit-detector_metropolis-hastings_2026-05-20.md) |
| 43 | computational-methods/numerical-optimization | 15% | DILUTED | no | [file](computational-methods/numerical-optimization/bullshit-detector_numerical-optimization_2026-05-20.md) |
| 44 | computational-methods/particle-filter | 25% | DILUTED | no | [file](computational-methods/particle-filter/bullshit-detector_particle-filter_2026-05-20.md) |
| 45 | computational-methods/perturbation-linearization | 20% | MISLABELED | no | [file](computational-methods/perturbation-linearization/bullshit-detector_perturbation-linearization_2026-05-20.md) |
| 46 | computational-methods/projection-methods | 10% | DILUTED | no | [file](computational-methods/projection-methods/bullshit-detector_projection-methods_2026-05-20.md) |
| 47 | computational-methods/simulation-based-estimation | 20% | DILUTED | yes | [file](computational-methods/simulation-based-estimation/bullshit-detector_simulation-based-estimation_2026-05-20.md) |
| 48 | computational-methods/smolyak-sparse-grids | 30% | DATA DRIFT | no | [file](computational-methods/smolyak-sparse-grids/bullshit-detector_smolyak-sparse-grids_2026-05-20.md) |
| 49 | structural-econometrics/adversarial-estimation | 25% | FALSE | no | [file](structural-econometrics/adversarial-estimation/bullshit-detector_adversarial-estimation_2026-05-20.md) |
| 50 | structural-econometrics/auction-valuation-recovery | 30% | DILUTED | no | [file](structural-econometrics/auction-valuation-recovery/bullshit-detector_auction-valuation-recovery_2026-05-20.md) |
| 51 | structural-econometrics/bayesian-dsge-hmc | 35% | DILUTED | yes | [file](structural-econometrics/bayesian-dsge-hmc/bullshit-detector_bayesian-dsge-hmc_2026-05-20.md) |
| 52 | structural-econometrics/dcegm-retirement-saving | 20% | DILUTED | no | [file](structural-econometrics/dcegm-retirement-saving/bullshit-detector_dcegm-retirement-saving_2026-05-20.md) |
| 53 | structural-econometrics/keane-wolpin-career-choice | 35% | FALSE | no | [file](structural-econometrics/keane-wolpin-career-choice/bullshit-detector_keane-wolpin-career-choice_2026-05-20.md) |
| 54 | structural-econometrics/q-learning-bus-engine | 25% | DILUTED | yes | [file](structural-econometrics/q-learning-bus-engine/bullshit-detector_q-learning-bus-engine_2026-05-20.md) |
| 55 | structural-econometrics/rum-choice-networks | 0% | HOLDS | no | [file](structural-econometrics/rum-choice-networks/bullshit-detector_rum-choice-networks_2026-05-20.md) |
| 56 | numerical-methods/bayesian-optimization | 30% | FALSE | no | [file](numerical-methods/bayesian-optimization/bullshit-detector_bayesian-optimization_2026-05-20.md) |
| 57 | numerical-methods/constrained-optimization-kkt | 25% | FALSE | no | [file](numerical-methods/constrained-optimization-kkt/bullshit-detector_constrained-optimization-kkt_2026-05-20.md) |
| 58 | numerical-methods/fixed-point-acceleration | 35% | FALSE | yes | [file](numerical-methods/fixed-point-acceleration/bullshit-detector_fixed-point-acceleration_2026-05-20.md) |
| 59 | numerical-methods/global-search-multistart | 75% | FALSE | yes | [file](numerical-methods/global-search-multistart/bullshit-detector_global-search-multistart_2026-05-20.md) |
| 60 | numerical-methods/interpolation | 50% | FALSE | yes | [file](numerical-methods/interpolation/bullshit-detector_interpolation_2026-05-20.md) |
| 61 | numerical-methods/root-finding | 25% | FALSE | no | [file](numerical-methods/root-finding/bullshit-detector_root-finding_2026-05-20.md) |
| 62 | numerical-methods/scalar-optimization-monopoly-pricing | 25% | DATA DRIFT | no | [file](numerical-methods/scalar-optimization-monopoly-pricing/bullshit-detector_scalar-optimization-monopoly-pricing_2026-05-20.md) |
| 63 | heterogeneous-agents/aiyagari-hact | 40% | FALSE | yes | [file](heterogeneous-agents/aiyagari-hact/bullshit-detector_aiyagari-hact_2026-05-20.md) |
| 64 | heterogeneous-agents/endogenous-grid-points | 25% | FALSE | no | [file](heterogeneous-agents/endogenous-grid-points/bullshit-detector_endogenous-grid-points_2026-05-20.md) |
| 65 | heterogeneous-agents/envelope-equation-iteration | 30% | MISLABELED | yes | [file](heterogeneous-agents/envelope-equation-iteration/bullshit-detector_envelope-equation-iteration_2026-05-20.md) |
| 66 | heterogeneous-agents/huggett-aggregate-risk-srl | 55% | FALSE | yes | [file](heterogeneous-agents/huggett-aggregate-risk-srl/bullshit-detector_huggett-aggregate-risk-srl_2026-05-20.md) |
| 67 | heterogeneous-agents/huggett-incomplete-markets | 20% | DATA DRIFT | no | [file](heterogeneous-agents/huggett-incomplete-markets/bullshit-detector_huggett-incomplete-markets_2026-05-20.md) |
| 68 | heterogeneous-agents/sequence-space-jacobian-hank | 20% | MISLABELED | no | [file](heterogeneous-agents/sequence-space-jacobian-hank/bullshit-detector_sequence-space-jacobian-hank_2026-05-20.md) |
| 69 | game-theory/cfr-asymmetric-auction | 15% | DILUTED | no | [file](game-theory/cfr-asymmetric-auction/bullshit-detector_cfr-asymmetric-auction_2026-05-20.md) |
| 70 | game-theory/deep-optimal-auctions | 15% | MISLABELED | no | [file](game-theory/deep-optimal-auctions/bullshit-detector_deep-optimal-auctions_2026-05-20.md) |
| 71 | game-theory/first-price-auctions | 15% | DILUTED | no | [file](game-theory/first-price-auctions/bullshit-detector_first-price-auctions_2026-05-20.md) |
| 72 | game-theory/normal-form-games | 10% | DILUTED | no | [file](game-theory/normal-form-games/bullshit-detector_normal-form-games_2026-05-20.md) |
| 73 | game-theory/quantal-response-equilibrium | 5% | HOLDS | no | [file](game-theory/quantal-response-equilibrium/bullshit-detector_quantal-response-equilibrium_2026-05-20.md) |
| 74 | game-theory/static-games | 10% | DILUTED | no | [file](game-theory/static-games/bullshit-detector_static-games_2026-05-20.md) |
| 75 | time-series/ar-processes | 20% | DATA DRIFT | no | [file](time-series/ar-processes/bullshit-detector_ar-processes_2026-05-20.md) |
| 76 | time-series/fred-macro-data | 15% | DILUTED | no | [file](time-series/fred-macro-data/bullshit-detector_fred-macro-data_2026-05-20.md) |
| 77 | time-series/minnesota-svar | 15% | DATA DRIFT | no | [file](time-series/minnesota-svar/bullshit-detector_minnesota-svar_2026-05-20.md) |
| 78 | time-series/ridge-lasso-sparsity | 20% | DILUTED | no | [file](time-series/ridge-lasso-sparsity/bullshit-detector_ridge-lasso-sparsity_2026-05-20.md) |
| 79 | time-series/stock-watson | 35% | DILUTED | yes | [file](time-series/stock-watson/bullshit-detector_stock-watson_2026-05-20.md) |
| 80 | agent-based-models/algorithmic-collusion-q-learning | 15% | DILUTED | no | [file](agent-based-models/algorithmic-collusion-q-learning/bullshit-detector_algorithmic-collusion-q-learning_2026-05-20.md) |
| 81 | agent-based-models/brock-hommes-asset-pricing | 15% | DILUTED | no | [file](agent-based-models/brock-hommes-asset-pricing/bullshit-detector_brock-hommes-asset-pricing_2026-05-20.md) |
| 82 | agent-based-models/cobweb-arifovic-ga-learning | 15% | DILUTED | no | [file](agent-based-models/cobweb-arifovic-ga-learning/bullshit-detector_cobweb-arifovic-ga-learning_2026-05-20.md) |
| 83 | agent-based-models/schelling-segregation | 5% | DATA DRIFT | no | [file](agent-based-models/schelling-segregation/bullshit-detector_schelling-segregation_2026-05-20.md) |
| 84 | agent-based-models/zero-intelligence-traders | 0% | HOLDS | no | [file](agent-based-models/zero-intelligence-traders/bullshit-detector_zero-intelligence-traders_2026-05-20.md) |
| 85 | global-dsge/deep-learning-optimal-growth | 20% | DILUTED | no | [file](global-dsge/deep-learning-optimal-growth/bullshit-detector_deep-learning-optimal-growth_2026-05-20.md) |
| 86 | global-dsge/heaton-lucas | 25% | DILUTED | no | [file](global-dsge/heaton-lucas/bullshit-detector_heaton-lucas_2026-05-20.md) |
| 87 | global-dsge/rbc-capital-tax | 15% | DILUTED | no | [file](global-dsge/rbc-capital-tax/bullshit-detector_rbc-capital-tax_2026-05-20.md) |
| 88 | global-dsge/rbc-irreversible-investment | 20% | MISLABELED | no | [file](global-dsge/rbc-irreversible-investment/bullshit-detector_rbc-irreversible-investment_2026-05-20.md) |
| 89 | dsge/assetNews | 10% | MISLABELED | no | [file](dsge/assetNews/bullshit-detector_assetNews_2026-05-20.md) |
| 90 | dsge/behavioral-nk | 25% | DILUTED | yes | [file](dsge/behavioral-nk/bullshit-detector_behavioral-nk_2026-05-20.md) |
| 91 | dsge/nkdsge | 5% | HOLDS | no | [file](dsge/nkdsge/bullshit-detector_nkdsge_2026-05-20.md) |
| 92 | dsge/rbc | 20% | DATA DRIFT | no | [file](dsge/rbc/bullshit-detector_rbc_2026-05-20.md) |
| 93 | optimal-control/hjb-growth | 15% | DATA DRIFT | no | [file](optimal-control/hjb-growth/bullshit-detector_hjb-growth_2026-05-20.md) |
| 94 | optimal-control/phase-diagrams | 25% | FALSE | no | [file](optimal-control/phase-diagrams/bullshit-detector_phase-diagrams_2026-05-20.md) |
| 95 | optimal-control/ramsey-growth | 20% | DATA DRIFT | no | [file](optimal-control/ramsey-growth/bullshit-detector_ramsey-growth_2026-05-20.md) |
| 96 | spatial-economics/allen-arkolakis | 15% | DATA DRIFT | no | [file](spatial-economics/allen-arkolakis/bullshit-detector_allen-arkolakis_2026-05-20.md) |
| 97 | bayesian-methods/neural-posterior-brock-hommes | 35% | FALSE | yes | [file](bayesian-methods/neural-posterior-brock-hommes/bullshit-detector_neural-posterior-brock-hommes_2026-05-20.md) |
