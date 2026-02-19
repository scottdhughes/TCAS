# TCAS Credence Update Math (Supplementary)

This note defines the full credence-update equations referenced in the camera-ready paper.

## Scope note
These equations define how TCAS computes theory-indexed credence bands **when all required streams are available**. In the empirical GPT-5.2 Pro walkthrough in this paper, O-stream data were not collected; under the missing-stream rule, credence bands are therefore withheld.

## Variable glossary
- `t`: consciousness-theory index.
- `s`: evidence-stream index (`B`, `M`, `P`, `O`).
- `x_{s,t}`: normalized support from stream `s` for theory `t`, in `[0,1]`.
- `n_s`: stream weight for stream `s`.
- `R^2_cue`: observer-confound cue-explained variance estimate.
- `rho`: baseline stream-correlation parameter in `[0,1]`.
- `o`: overlap factor in `[0,1]` for shared-channel evidence.
- `rho_eff`: overlap-adjusted correlation penalty.
- `n_eff`: effective evidence weight after overlap discount.
- `alpha_t, beta_t`: prior Beta parameters for theory `t`.
- `alpha'_t, beta'_t`: posterior Beta parameters for theory `t`.
- `xbar_t`: weighted mean support for theory `t`.

## Stream preprocessing
Behavioral weight is adjusted for observer confounds:

`n_B <- (1 - R^2_cue) * n_B`

## Overlap discount
Overlap-adjusted penalty:

`rho_eff = rho * (1 - 0.5 * o)`

Effective total weight:

`n_eff = (1 - rho_eff) * sum_s n_s`

## Posterior update
Weighted support mean:

`xbar_t = (sum_s n_s * x_{s,t}) / (sum_s n_s)`

Given prior `z_t ~ Beta(alpha_t, beta_t)`, posterior parameters are:

`alpha'_t = alpha_t + n_eff * xbar_t`

`beta'_t  = beta_t  + n_eff * (1 - xbar_t)`

Credence band reported in TCAS cards: 10th to 90th percentile of `Beta(alpha'_t, beta'_t)`.

## Worked example (illustrative)
Use prior `Beta(1,4)`, stream supports and weights:
- `x_B = 0.80`, `n_B = 5`
- `x_P = 0.90`, `n_P = 4`
- `R^2_cue = 0.30`
- `rho = 0.20`, `o = 0` (so `rho_eff = 0.20`)

1) Down-weight B:

`n_B <- (1 - 0.30) * 5 = 3.5`

2) Effective weight:

`n_eff = (1 - 0.20) * (3.5 + 4) = 0.8 * 7.5 = 6.0`

3) Weighted support:

`xbar = (3.5*0.80 + 4*0.90) / 7.5 = 6.4 / 7.5 = 0.8533`

4) Posterior:

`alpha' = 1 + 6.0*0.8533 = 6.12`

`beta'  = 4 + 6.0*(1 - 0.8533) = 4.88`

Posterior Beta(6.12, 4.88) yields the reported mid-range credence profile.

## Code pointer
Reference implementation: `code/tcas/aggregation.py`.
