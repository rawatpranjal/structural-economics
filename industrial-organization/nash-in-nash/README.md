# Hospital-Insurer Network Bargaining with Nash-in-Nash

> How network outside options shape negotiated hospital payments.

## Overview

Health insurers sell networks, not isolated hospital contracts. A plan that loses a high-quality hospital becomes less attractive to enrollees, and a plan that loses a whole hospital system can become nearly unusable. The tutorial computes how those outside options translate into negotiated hospital payments.

Nash-in-Nash supplies the pricing rule for the contracts. For each hospital-insurer link, the calculation removes that link, recomputes insurer demand, converts lost enrollment into downstream surplus, and splits the surplus by the bargaining weight. The network is small enough to enumerate every disagreement case, but the object is the same one used in empirical IO work on hospital systems, cable channels, and other vertical markets with many bilateral negotiations.

The example keeps premiums fixed so the computation isolates network bargaining. The [vertical relationships](../vertical-relationships/) tutorial studies double marginalization in a simpler channel model, while [merger simulation](../merger-simulation/) turns changed ownership into final-price counterfactuals.

## Equations

Let $d \in \{1,\ldots,D\}$ index insurers and $h \in \{1,\ldots,H\}$ index
hospitals. Insurer $d$ has hospital network $G_d$. In the full agreement
network $G$, each insurer carries both hospitals.

Demand is a logit over insurers and an outside option:

$$
q_d(G) =
M \frac{\exp(v_d(G_d) / \sigma_\varepsilon)}
{1 + \sum_{\ell=1}^{D} \exp(v_\ell(G_\ell) / \sigma_\varepsilon)} .
$$

The deterministic utility of an insurer is

$$
v_d(G_d) = Q(G_d) - P_d,
$$

where $P_d$ is the premium. The network value is

$$
Q(\emptyset)=0,\qquad
Q(G_d)=\max_{h \in G_d} a_h + \eta(|G_d|-1)
\quad\text{when }G_d\neq\emptyset .
$$

Here $a_h$ is hospital quality and $\eta$ is the incremental value of a second
in-network hospital. Let $m_d=P_d-c_d^D$ be the insurer margin before hospital
transfers. If the link $(h,d)$ fails, the disagreement network is $G^{-hd}$.
The gross incremental value of hospital $h$ to insurer $d$ is

$$
\Delta_{hd}=m_d\left[q_d(G)-q_d(G^{-hd})\right].
$$

The bilateral surplus net of hospital cost $c_h^H$ is

$$
S_{hd}=\Delta_{hd}-c_h^H q_d(G).
$$

The Nash bargain over the per-enrollee hospital transfer $w_{hd}$ solves

$$
\max_{w_{hd}}
\left[(w_{hd}-c_h^H)q_d(G)\right]^\tau
\times
\left[\Delta_{hd}-w_{hd}q_d(G)\right]^{1-\tau},
$$

so the transfer is

$$
w_{hd}=c_h^H + \tau \frac{S_{hd}}{q_d(G)}
      =(1-\tau)c_h^H+\tau\frac{\Delta_{hd}}{q_d(G)} .
$$

For a merged hospital system $H$, the relevant disagreement removes all system
hospitals from insurer $d$. With $C_H=\sum_h c_h^H$,

$$
W_{Hd}=C_H+\tau\frac{
m_d[q_d(G)-q_d(G^{-Hd})]-C_Hq_d(G)
}{q_d(G)}
$$

is the system-level per-enrollee transfer.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Hospitals | 2 | Upstream negotiators |
| Insurers | 2 | Downstream plans selling to consumers |
| Market size $M$ | 1000 | Potential enrollees |
| Bargaining weight $\tau$ | 0.50 | Hospital share of bilateral surplus |
| Hospital qualities $a_h$ | 20.0, 18.0 | Network utility shifters |
| Hospital costs $c_h^H$ | 1.0, 1.2 | Cost per enrolled member |
| Insurer premiums $P_d$ | 8.0, 8.5 | Fixed downstream prices |
| Insurer costs $c_d^D$ | 1.0, 1.0 | Non-hospital marginal costs |
| Second-hospital value $\eta$ | 3.0 | Extra network value beyond the best hospital |
| Logit scale $\sigma_\varepsilon$ | 5.0 | Controls substitution across insurers |

## Solution Method

The numerical work is enumeration plus a closed-form Nash split. The model does not solve a new premium equilibrium. It asks what each side would lose under a specific disagreement network, holding the other links fixed when the bargain is bilateral. Under system ownership, the disagreement removes the whole hospital system from the insurer's network.

```text
Algorithm: Nash-in-Nash transfers in a hospital-insurer network
Input: full networks G, premiums P, costs c^D and c^H, demand q(.), weight tau
Output: bilateral transfers w_hd and merged-system transfers W_Hd
Compute full-agreement demand q_d(G) for every insurer d
for each hospital h and insurer d:
    form G^{-hd} by removing hospital h only from insurer d
    compute disagreement demand q_d(G^{-hd})
    Delta_hd = (P_d - c_d^D) * [q_d(G) - q_d(G^{-hd})]
    S_hd = Delta_hd - c_h^H * q_d(G)
    w_hd = c_h^H + tau * S_hd / q_d(G)
for each insurer d under hospital-system ownership:
    form G^{-Hd} by removing the whole hospital system from insurer d
    compute the system surplus using the same demand object
    W_Hd = system cost + tau * system surplus / q_d(G)
```

At the baseline $\tau=0.50$, Hospital 1 receives higher transfers because its removal lowers network value more. Insurer 2 pays somewhat more because its higher premium gives it a larger per-enrollee margin to lose when the network deteriorates.

## Results

The bilateral transfers come from the enrollment loss created by a specific broken link. Dropping Hospital 1 hurts the insurer more than dropping Hospital 2 because Hospital 1 has higher network value. Insurer 2's higher premium also raises the dollar value of lost enrollment.

The left panel shows per-enrollee transfers with hospital costs marked as black points. The right panel shows the enrollment loss in each pair's disagreement network.

<img src="figures/negotiated-prices.png" alt="Bilateral hospital-insurer transfers and disagreement demand losses" width="80%">

Changing $\tau$ holds demand and disagreement networks fixed, so it changes only the division of surplus. Hospital profits rise with the bargaining weight. Insurer profits fall because a larger share of the same incremental network value is paid upstream.

The vertical guide marks the baseline calibration. The exercise is a surplus split, not a new demand equilibrium at each value of tau.

<img src="figures/profits-vs-bargaining.png" alt="Surplus division as the hospital bargaining weight changes" width="80%">

The merger comparison adds the two separate hospital transfers and compares that sum with a single system payment. In this calibration, either hospital alone keeps an insurer's network viable. Losing the merged system leaves the insurer with no in-network hospital, so the system-level disagreement payoff is lower and the negotiated payment is higher.

The comparison is at the insurer level because a merged system bargains over one total transfer, not two independent hospital prices.

<img src="figures/merger-prices.png" alt="Separate hospital payments versus merged-system payment" width="80%">

The table reports the quantities used in the Nash bargain. Gross value is the downstream margin times the enrollment loss; surplus subtracts hospital cost.

**Bilateral Bargaining Diagnostics**

| Pair                   |   Full demand |   Disagreement demand |   Demand loss |   Gross value / enrollee |   Surplus / enrollee |   Hospital cost |   Transfer |
|:-----------------------|--------------:|----------------------:|--------------:|-------------------------:|---------------------:|----------------:|-----------:|
| Hospital 1 - Insurer 1 |         511.6 |                 278.2 |         233.4 |                    3.194 |                2.194 |             1   |      2.097 |
| Hospital 1 - Insurer 2 |         462.9 |                 240.7 |         222.2 |                    3.6   |                2.6   |             1   |      2.3   |
| Hospital 2 - Insurer 1 |         511.6 |                 365   |         146.6 |                    2.005 |                0.805 |             1.2 |      1.603 |
| Hospital 2 - Insurer 2 |         462.9 |                 321.1 |         141.8 |                    2.297 |                1.097 |             1.2 |      1.749 |

The merged-system rows use a different disagreement event from the bilateral rows: the insurer loses both hospitals at once.

**Ownership Counterfactual**

| Insurer   |   Full demand |   Demand without system |   Separate hospital transfers |   Merged system transfer |   Change (%) |
|:----------|--------------:|------------------------:|------------------------------:|-------------------------:|-------------:|
| Insurer 1 |         511.6 |                    10.4 |                         3.7   |                    4.529 |         22.4 |
| Insurer 2 |         462.9 |                     8.6 |                         4.048 |                    4.78  |         18.1 |

## Takeaway

Nash-in-Nash bargaining maps a vertical contract into an observable counterfactual: what does the insurer lose if this agreement fails while the rest of the contracting network remains intact? The answer depends on substitution across insurers, the hospital's incremental network value, and the ownership structure that defines the relevant outside option.

## References

- Horn, H. and Wolinsky, A. (1988). "Bilateral Monopolies and Incentives for Merger." *RAND Journal of Economics*, 19(3).
- Crawford, G. and Yurukoglu, A. (2012). "The Welfare Effects of Bundling in Multichannel Television Markets." *American Economic Review*, 102(2).
- Ho, K. and Lee, R. (2017). "Insurer Competition in Health Care Markets." *Econometrica*, 85(2).
