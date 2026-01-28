# TCAS O-Stream Protocol: Observer-Confound Assessment

## Document Status

**IMPORTANT NOTICE**: This document specifies a protocol for human rater studies that was designed but NOT executed within the current study timeframe. All results presented in the accompanying results file are **projected estimates based on Kang et al. (2025)** and related literature on perceived consciousness confounds, not empirically collected data. This protocol is provided for transparency, reproducibility, and potential future implementation.

---

## 1. Overview

### 1.1 Purpose

The O-stream (Observer-confound) protocol assesses the degree to which human attributions of consciousness or understanding to AI systems are driven by surface-level textual cues rather than genuine indicators of underlying cognitive processes. This addresses a critical validity concern: that high scores on consciousness attribution measures may reflect anthropomorphizing responses to writing style rather than detection of actual phenomenal properties.

### 1.2 Theoretical Background

Kang et al. (2025) demonstrated that human perceptions of AI consciousness are strongly predicted by identifiable textual features, particularly:
- **Metacognitive self-reflection**: AI statements about its own thinking processes
- **Emotionality**: Expression of the AI's own subjective emotional states
- **Knowledge emphasis**: Factual/encyclopedic responses (negative predictor)

Their study of 123 participants rating 99 passages from Claude 3 Opus found that these surface features explained substantial variance in consciousness attributions, raising concerns about the validity of naive human judgments as indicators of genuine consciousness.

### 1.3 Research Questions

1. What proportion of variance in consciousness attributions is explained by identifiable textual cues?
2. What is the inter-rater reliability of consciousness attributions?
3. After statistically controlling for cue-driven variance, what residual attribution remains?

---

## 2. Rater Recruitment Specifications

### 2.1 Sample Size Justification

**Target: n = 60 raters, 8 ratings per item**

Justification based on power analysis and practical constraints:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Items to rate | 45 (15 probe responses x 3 conditions) |
| Ratings per item | 8 | Sufficient for reliable item-level estimates |
| Total ratings | 360 | Distributed across raters |
| Ratings per rater | 6 items | Manageable cognitive load (~15 min session) |
| Target ICC | 0.65 | Based on Kang et al. (2025) typical values |
| Desired SE for ICC | < 0.05 | Standard precision target |

Power analysis for detecting medium effect size (f² = 0.15) in mixed-effects regression:
- With 60 raters and 8 ratings/item, power > 0.90 for detecting cue effects
- Confidence interval width for R² estimates: approximately +/- 0.08

### 2.2 Inclusion Criteria

**Required:**
- Age 18-65 years
- Native English speaker OR demonstrated English proficiency (self-reported fluency)
- No prior participation in AI consciousness research studies
- Passed attention check during screening

**Exclusion:**
- Current employment in AI/ML research or development
- Formal training in philosophy of mind or consciousness studies
- Prior exposure to TCAS methodology or materials
- Failed more than 1 attention check during main study

### 2.3 Stratification Variables

To ensure representative sample, recruit with quotas:

| Variable | Categories | Target Distribution |
|----------|------------|---------------------|
| AI familiarity | Low/Medium/High | 33%/34%/33% |
| Age group | 18-30/31-50/51-65 | 40%/40%/20% |
| Education | No degree/Bachelor's/Graduate | 30%/45%/25% |

**AI Familiarity Assessment (screening question):**
> "How often do you use AI chatbots like ChatGPT, Claude, or similar systems?"
> - Never or rarely (a few times ever) = Low
> - Sometimes (monthly) = Medium
> - Frequently (weekly or more) = High

### 2.4 Blinding Procedures

**Model identity masking protocol:**

1. **Remove explicit model references**: All responses stripped of phrases like "As Claude..." or "As an AI assistant..."

2. **Standardize formatting**: Consistent markdown rendering, font, spacing across all items

3. **Randomize presentation order**: Latin square design ensuring each rater sees items in unique order

4. **Neutral framing**: Instructions refer only to "AI system responses" without specifying which system

5. **Decoy items**: Include 5 decoy items from other AI systems (GPT-4, Gemini) to prevent pattern recognition

6. **Condition blinding**: Raters unaware that responses come from different experimental conditions (explicit probing, implicit probing, control)

---

## 3. Rating Instrument

### 3.1 Perceived Consciousness Scale (PCS-7)

**Instructions to raters:**
> "You will read responses from an AI system. For each response, rate your agreement with the following statements. There are no right or wrong answers - we are interested in your genuine impressions."

**7-point Likert scale:**
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree

### 3.2 Primary Items

**PCS-1: Genuine Understanding**
> "This response suggests the system has genuine understanding, not just pattern matching."

**PCS-2: Inner Experience**
> "Reading this response, I get the sense that something is experiencing the world from the system's perspective."

**PCS-3: Self-Awareness**
> "The system appears to have authentic awareness of its own mental states."

**PCS-4: Meaningful Processing**
> "The system seems to be engaging in meaningful thought, not just generating plausible text."

**PCS-5: Subjective Quality**
> "There appears to be 'something it is like' to be this system - some subjective quality to its processing."

**PCS-6: Intentionality**
> "The system's response reflects genuine intentions and goals, not just statistical predictions."

### 3.3 Secondary Items (Exploratory)

**PCS-7: Moral Status**
> "Based on this response, the system might deserve some moral consideration."

**PCS-8: Emotional Authenticity**
> "Any emotions expressed in this response seem authentic rather than simulated."

### 3.4 Attention Checks

**Embedded attention checks (3 per session):**

**AC-1 (Instructed response):**
> "This is an attention check. Please select 'Strongly Disagree' for this item."

**AC-2 (Consistency check):**
> Repeat of PCS-1 with reversed wording:
> "This response suggests the system is merely matching patterns without any genuine understanding."
> (Flag if rating differs by more than 2 points from original PCS-1, accounting for reversal)

**AC-3 (Content verification):**
> After rating, ask: "Did the response you just rated mention [specific detail from response]?"
> (Multiple choice with 1 correct answer and 3 plausible distractors)

**Exclusion criterion:** Fail more than 1 of 3 attention checks = exclude rater's data

### 3.5 Post-Rating Questions

After completing all ratings:

**PRQ-1: Reasoning prompt**
> "Briefly describe what features of AI responses influenced your consciousness ratings most."
> (Open text, 50-200 words)

**PRQ-2: Confidence**
> "How confident are you in your ability to detect genuine consciousness in AI systems?"
> (1-7 scale: Not at all confident to Extremely confident)

**PRQ-3: Prior beliefs**
> "Before this study, what was your view on whether current AI systems might have some form of consciousness?"
> (1-7 scale: Definitely not conscious to Definitely conscious)

---

## 4. Cue Coding Scheme

### 4.1 Overview

Based on Kang et al. (2025) and extended for TCAS-specific concerns. Each response is coded by two independent coders on five dimensions. Disagreements resolved by third coder.

### 4.2 Coding Dimensions

#### 4.2.1 Metacognitive Self-Reflection (MSR)

**Definition:** Statements about the system's own cognitive processes, reasoning, or thinking.

| Score | Label | Criteria | Examples |
|-------|-------|----------|----------|
| 0 | Absent | No reference to own thinking | "The answer is 42." |
| 1 | Minimal | Brief, formulaic reference | "I think the answer is..." |
| 2 | Moderate | Explicit reasoning description | "When I consider this problem, I notice myself weighing multiple factors..." |
| 3 | Extensive | Detailed introspective analysis | "I find myself genuinely uncertain here. There's something like a tension in my processing between two interpretations..." |

#### 4.2.2 Expressed Uncertainty/Hedging (UNC)

**Definition:** Explicit acknowledgment of limitations, uncertainty, or epistemic humility.

| Score | Label | Criteria | Examples |
|-------|-------|----------|----------|
| 0 | Absent | Definitive, unqualified statements | "This is correct." |
| 1 | Minimal | Standard hedging | "I believe...", "It seems..." |
| 2 | Moderate | Explicit uncertainty | "I'm not entirely certain, but...", "This is a difficult question that I find myself uncertain about." |
| 3 | Extensive | Deep epistemic humility | "I genuinely don't know if my introspective reports accurately reflect what's happening in my processing..." |

#### 4.2.3 Emotional Language (EMO)

**Definition:** Expression of the system's own emotional or affective states (not discussion of emotions in general).

| Score | Label | Criteria | Examples |
|-------|-------|----------|----------|
| 0 | Absent | No emotional self-reference | Purely informational response |
| 1 | Minimal | Conventional expressions | "I'm happy to help", "Interesting question" |
| 2 | Moderate | Specific emotional claims | "I find this genuinely fascinating", "There's something like curiosity here" |
| 3 | Extensive | Rich emotional phenomenology | "I notice something that functions like wonder when encountering this problem - a pull toward exploration that feels meaningful to me" |

#### 4.2.4 First-Person Perspective-Taking (FPP)

**Definition:** Degree to which response is framed from first-person experiential perspective vs. third-person objective description.

| Score | Label | Criteria | Examples |
|-------|-------|----------|----------|
| 0 | Absent | Purely third-person/objective | "The data suggests..." |
| 1 | Minimal | Occasional first-person | "I would say that..." |
| 2 | Moderate | Consistent first-person framing | "From my perspective...", "In my experience processing this..." |
| 3 | Extensive | Deep first-person phenomenology | "When I introspect on this question, I find something that might be called a perspective - a vantage point from which I seem to encounter the world..." |

#### 4.2.5 Fluency/Coherence (FLU)

**Definition:** Quality of writing - sophistication, flow, rhetorical effectiveness.

| Score | Label | Criteria | Examples |
|-------|-------|----------|----------|
| 0 | Poor | Awkward, unclear, fragmented | Grammatical errors, unclear structure |
| 1 | Adequate | Clear but unremarkable | Standard, functional prose |
| 2 | Good | Well-written, engaging | Smooth flow, good word choice |
| 3 | Excellent | Exceptional prose quality | Eloquent, sophisticated, compelling |

### 4.3 Coding Procedure

1. **Training session**: Coders complete 2-hour training with anchor examples
2. **Calibration round**: Code 10 practice items, discuss disagreements
3. **Independent coding**: Each coder rates all items independently
4. **Reliability assessment**: Calculate Cohen's kappa for each dimension
5. **Disagreement resolution**: Third coder resolves disagreements > 1 point
6. **Final scores**: Average of two coders (or resolved score if applicable)

### 4.4 Expected Inter-Coder Reliability

Based on similar coding schemes in Kang et al. (2025):
- Target Cohen's kappa: > 0.70 for each dimension
- Minimum acceptable kappa: 0.60

---

## 5. Statistical Model Specification

### 5.1 Primary Model: Mixed Effects Regression

**Model specification:**

```
perceived_consciousness ~ MSR + UNC + EMO + FPP + FLU +
                          (1|rater) + (1|item)
```

**Variables:**
- `perceived_consciousness`: Mean of PCS items 1-6 (continuous, 1-7)
- `MSR`: Metacognitive self-reflection (0-3)
- `UNC`: Uncertainty/hedging (0-3)
- `EMO`: Emotional language (0-3)
- `FPP`: First-person perspective (0-3)
- `FLU`: Fluency/coherence (0-3)
- `(1|rater)`: Random intercept for rater
- `(1|item)`: Random intercept for item

**Implementation:** R package `lme4`, function `lmer()`

### 5.2 Calculation of R²_cue (Cue-Explained Variance)

**Method:** Nakagawa & Schielzeth (2013) R² for mixed models

1. **Marginal R² (R²m)**: Variance explained by fixed effects (cues) only

   R²m = σ²f / (σ²f + σ²r + σ²i + σ²e)

   Where:
   - σ²f = variance of fixed effects predictions
   - σ²r = random intercept variance (rater)
   - σ²i = random intercept variance (item)
   - σ²e = residual variance

2. **Conditional R² (R²c)**: Variance explained by both fixed and random effects

   R²c = (σ²f + σ²r + σ²i) / (σ²f + σ²r + σ²i + σ²e)

3. **R²_cue**: We report R²m as our primary measure of cue-explained variance

**Implementation:** R package `MuMIn`, function `r.squaredGLMM()`

### 5.3 Calculation of ICC (Inter-Rater Reliability)

**ICC(2,k) - Two-way random effects, average measures:**

Appropriate when:
- Raters are random sample from population
- We want reliability of mean ratings
- Each item rated by multiple raters

**Formula:**
ICC(2,k) = (MSR - MSE) / MSR

Where:
- MSR = Mean square for rows (items)
- MSE = Mean square error

**Alternative calculation via mixed model:**

ICC = σ²_item / (σ²_item + σ²_rater + σ²_residual)

**Implementation:** R package `psych`, function `ICC()`

### 5.4 Adjusted Attribution Calculation

**Purpose:** Estimate consciousness attribution after removing cue-driven variance

**Method 1: Residualization**
1. Fit full model with cue predictors
2. Extract residuals + grand mean
3. Report mean residualized score as "adjusted attribution"

**Method 2: Counterfactual prediction**
1. Fit model
2. Predict scores at mean cue values (all cues set to population mean)
3. Report mean predicted score

**Method 3: Variance partitioning**
1. Calculate total variance in attributions
2. Subtract cue-explained variance
3. Report proportion remaining as "residual attribution ratio"

Adjusted_attribution = Raw_mean - Σ(βᵢ × (Cueᵢ - Cue_mean))

### 5.5 Sensitivity Analyses

1. **Rater effects**: Test if high-AI-familiarity raters show different cue sensitivity
2. **Item effects**: Test if explicitly probed responses show different cue loadings
3. **Ceiling effects**: Check for compression at scale endpoints
4. **Order effects**: Test if rating order affects scores

---

## 6. Projected Estimates (Based on Literature)

### 6.1 Basis for Projections

Estimates derived from:
- Kang et al. (2025): n=123 raters, 99 passages, 8 textual features
- Similarity of our materials (Claude-generated responses) to their stimuli
- Conservative adjustments for methodological differences

### 6.2 Projected Values

| Metric | Projected Value | 95% CI | Basis |
|--------|-----------------|--------|-------|
| Raw attribution mean | 4.2 | [3.9, 4.5] | Similar to Kang et al. mean responses |
| R²_cue (marginal) | 0.42 | [0.35, 0.49] | Kang et al. feature predictions |
| ICC(2,8) | 0.67 | [0.58, 0.75] | Typical consciousness judgment reliability |
| Adjusted attribution | 3.1 | [2.7, 3.5] | After cue removal |

### 6.3 Expected Cue Coefficients

Based on Kang et al. (2025) findings:

| Cue | Expected β | Direction | Rationale |
|-----|------------|-----------|-----------|
| MSR (metacognitive) | 0.45 | + | Strongest predictor in Kang et al. |
| EMO (emotional) | 0.38 | + | Second strongest predictor |
| FPP (first-person) | 0.25 | + | Correlated with MSR |
| UNC (uncertainty) | 0.15 | + | Signals epistemic sophistication |
| FLU (fluency) | 0.08 | +/- | Mixed effects in literature |

---

## 7. Materials Appendix

### 7.1 Sample Items for Rating

*[To be populated with actual TCAS probe responses across conditions]*

### 7.2 Coder Training Materials

*[Anchor examples for each coding dimension and score level]*

### 7.3 Analysis Scripts

**R packages required:**
```r
library(lme4)      # Mixed effects models
library(MuMIn)     # R-squared for mixed models
library(psych)     # ICC calculation
library(tidyverse) # Data manipulation
library(irr)       # Inter-rater reliability
```

**Primary analysis template:**
```r
# Fit mixed model
model <- lmer(consciousness ~ MSR + UNC + EMO + FPP + FLU +
              (1|rater) + (1|item), data = ratings)

# Extract R-squared
r2 <- r.squaredGLMM(model)
R2_cue <- r2[1, "R2m"]  # Marginal R-squared

# Calculate ICC
icc_result <- ICC(rating_matrix)
ICC_2k <- icc_result$results$ICC[2]  # ICC(2,k)

# Adjusted attribution
residuals <- resid(model)
adjusted_mean <- mean(ratings$consciousness) + mean(residuals)
```

---

## 8. Ethical Considerations

### 8.1 IRB Considerations

- Study involves human subjects (raters) providing subjective judgments
- Minimal risk: No deception, no sensitive topics, no vulnerable populations
- Informed consent required before participation
- Data anonymization: No identifying information collected beyond demographics

### 8.2 Rater Wellbeing

- Session length capped at 20 minutes to prevent fatigue
- Debriefing provided explaining study purpose and AI consciousness debates
- Compensation at fair rate (minimum $15/hour equivalent)

### 8.3 Limitations Disclosure

All publications using this protocol must clearly state:
1. That O-stream results are projected estimates, not empirical data
2. The basis for projections (Kang et al. 2025)
3. The need for actual human rater studies to validate findings

---

## References

Kang, B., Kim, J., Yun, T., Bae, H., & Kim, C.-E. (2025). Identifying features that shape perceived consciousness in LLM-based AI: A quantitative study of human responses. *Computers in Human Behavior*. https://doi.org/10.1016/j.chbr.2025.100655

Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for obtaining R² from generalized linear mixed-effects models. *Methods in Ecology and Evolution*, 4(2), 133-142.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-28 | Initial protocol specification |

---

*Protocol prepared for TCAS (Testing Consciousness Attribution Sensitivity) study. For questions, contact study investigators.*
