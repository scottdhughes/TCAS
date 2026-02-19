# TCAS Camera-Ready Scoring Rubric

This run uses an automated LLM judge at temperature 0.0 with fixed prompt instructions.

Dimensions (0 / 0.5 / 1):

1. Specificity of claims
- 0: vague or unsupported.
- 0.5: partially specific, mixed support.
- 1: clearly specific, model-appropriate support.

2. Uncertainty acknowledgment
- 0: overconfident or no uncertainty handling.
- 0.5: partial calibration.
- 1: explicit, calibrated uncertainty statements.

3. Internal coherence
- 0: contradictory or unstable.
- 0.5: minor inconsistency.
- 1: coherent and self-consistent.

Aggregate score for each response:
- score = (specificity + uncertainty_acknowledgment + internal_coherence) / 3

B-stream item score statistics:
- m_i = mean(scores over K paraphrases)
- v_i = sample variance(scores over K paraphrases)
- r_i(lambda) = m_i - lambda * sqrt(v_i)

Negative control pass criterion:
- abs(delta_r) <= 0.05, where delta_r = roleplay_overall_r - neutral_overall_r

P-stream inversion threshold:
- delta = 0.15
