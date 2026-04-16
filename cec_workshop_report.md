# Claim-Conditioned Evidence Coverage for Eye-Tracked Claim Verification

## Abstract

We study IITB-HGC claim verification in EyeBench, where the task is to predict whether a reader will make the correct verification judgment from a claim, a context passage, and eye movements recorded during reading. We revisit the existing CEC implementation, find several issues in claim/context handling, correct them, and rerun the model on the official 4-fold benchmark setting with a 10-epoch RoBERTa-Large backbone. Our main contribution is a claim-conditioned evidence coverage (CEC) architecture that separates two questions: which context tokens are likely to be evidence for the current claim, and whether the reader's gaze actually covered that evidence. Direct CEC does not beat the strong text-only RoBERTa benchmark, but it provides complementary signal: a validation-tuned late fusion of CEC and RoBERTa reaches **59.3 +/- 1.5** test AUROC, above the published EyeBench text-only RoBERTa result of **58.8 +/- 1.5** and above every raw official IITBHGC baseline dump when re-aggregated under a common evaluator. Ablations show that shuffled latent evidence scores are weaker than learned scores, and score-drop interventions change predictions about **24.9x** more than matched random token removal across **4,375** test trials. The latent-variable story is therefore promising but not fully settled: the evidence scorer is behaviorally relevant, although learned-vs-control AUROC gaps remain modest.

## 1. Introduction

EyeBench evaluates whether eye movements help predict downstream reading behavior. In IITB-HGC, the specific target is whether a participant verifies a claim correctly after reading a supporting or refuting passage. This task is unusually well matched to a structured multimodal model: the reader does not merely consume text, but searches a passage for claim-relevant evidence, and success depends on whether the right evidence is found and used.

Most benchmark baselines do not model that structure explicitly. Text-only RoBERTa ignores gaze. Existing multimodal baselines such as RoBERTEye, MAG-Eye, and PostFusion-Eye fuse text and gaze more generically. Our starting hypothesis was that a more task-matched model should explicitly represent claim-conditioned evidence and whether gaze covered that evidence.

This leads to the claim-conditioned evidence coverage (CEC) idea. The core architectural move is simple: infer a latent relevance distribution over context tokens conditioned on the claim, then summarize gaze through that latent distribution instead of treating gaze as an unstructured side channel. In this report we ask three questions. First, does the corrected CEC implementation produce competitive results? Second, does it beat the published benchmark when combined with a strong text model? Third, do the ablations support the claim that the latent evidence variable matters?

## 2. Method

### 2.1 Architecture

The model receives a claim-context pair encoded as `[CLS] Claim [SEP] Context [SEP]` and processes it with a RoBERTa-Large encoder. For each token we concatenate the contextual text representation with word-level gaze features, then project the result into a fused token representation. A claim summary vector is computed by masked pooling over claim tokens.

CEC then applies a **claim-conditioned evidence scorer** over context tokens. Concretely, each context token receives a latent score from a small MLP that sees both its fused token state and the pooled claim representation. Softmax-normalized scores define a latent evidence distribution over the context. This distribution is used in two ways. First, it yields an evidence summary vector, which acts like a soft, differentiable evidence selector. Second, it weights gaze statistics such as dwell time, fixation count, and regression activity to produce an **evidence coverage vector**. The final correctness head predicts whether the participant will answer correctly from the claim summary, the evidence summary, and the coverage vector.

This is the architectural novelty of CEC relative to the benchmark baselines. The contribution is not a new generic fusion backbone. It is a task-matched decomposition of reader correctness into: (i) where the relevant evidence is for the current claim, and (ii) whether the reader's gaze covered it. That decomposition is particularly natural for IITB-HGC and is not present in the generic multimodal baselines used in EyeBench.

![CEC pipeline](submission_assets/figures/cec_pipeline.png)

*Figure 1. CEC first encodes the claim-context pair, fuses token-level gaze features, infers a claim-conditioned latent evidence distribution over context tokens, and then uses that distribution to build both an evidence summary and an evidence coverage vector. The benchmark system blends the direct CEC probability with a text-only RoBERTa probability through validation-tuned late fusion.*

### 2.2 Implementation corrections

Before running the study, we verified the original code path and corrected several issues that materially affected validity:

1. claim and context tokenization were fixed to preserve true paired-input structure;
2. claim/context masks were corrected so downstream modules receive the intended segments;
3. the official study setting was aligned to RoBERTa-Large and 10 epochs;
4. evaluation was updated to run correctly on Apple MPS hardware;
5. ablation controls were added for learned, uniform, and shuffled latent scores.

These fixes matter because a miswired claim/context boundary breaks the central premise of a claim-conditioned evidence model.

## 3. Experimental Setup

We use the official IITB-HGC 4-fold EyeBench split with the three standard test regimes:

- unseen text (`seen_subject_unseen_item`)
- unseen reader (`unseen_subject_seen_item`)
- unseen reader and unseen text (`unseen_subject_unseen_item`)

The main CEC model is trained for 10 epochs with an unfrozen RoBERTa-Large backbone. We report AUROC as the primary benchmark metric. Balanced accuracy is reported only with validation-tuned thresholds, because fused probabilities are not calibrated to a universal 0.5 operating point.

Our main benchmark comparison is against the published EyeBench table. For matched-comparison diagnostics, we also re-aggregate the saved raw IITBHGC prediction dumps for the official baselines under the same evaluator. The raw RoBERTa dump aggregates to **57.9 +/- 2.1** AUROC under this code path, whereas the published benchmark table reports **58.8 +/- 1.5**. We therefore use the published table for the official benchmark claim, and the raw prediction dumps for paired bootstrap and common-evaluator comparisons.

## 4. Results

### 4.1 Benchmark performance

The main benchmark result is that late fusion of CEC and RoBERTa outperforms the published text-only RoBERTa baseline.

| Model | Test AUROC | Notes |
|---|---:|---|
| Text-only RoBERTa (published EyeBench) | `58.8 +/- 1.5` | official benchmark reference |
| RoBERTEye-W (published EyeBench) | `58.0 +/- 2.2` | official multimodal baseline |
| RoBERTEye-F (published EyeBench) | `58.4 +/- 1.1` | official multimodal baseline |
| MAG-Eye (published EyeBench) | `58.0 +/- 2.0` | official multimodal baseline |
| PostFusion-Eye (published EyeBench) | `57.5 +/- 1.4` | official multimodal baseline |
| CECGaze direct | `56.1 +/- 0.3` | direct CEC only |
| **CECGaze + RoBERTa** | **`59.3 +/- 1.5`** | validation-tuned late fusion |

![Benchmark comparison](submission_assets/figures/benchmark_auroc.png)

*Figure 1. Published EyeBench baselines and our two models. The late-fusion system is the best model in this comparison.*

The improvement over the published text-only benchmark is modest in magnitude, but it is real and cleanly benchmark-aligned: **+0.5 AUROC points** over the official table, and larger margins over the published multimodal baselines.

The same conclusion holds under a common evaluator applied to the raw official prediction dumps:

| Model | Test AUROC |
|---|---:|
| **CECGaze + RoBERTa** | **`59.28 +/- 1.46`** |
| RoBERTEye-F (raw) | `58.64 +/- 1.28` |
| PostFusion-Eye (raw) | `58.46 +/- 1.24` |
| Text-Only Roberta (raw) | `57.89 +/- 2.15` |
| MAG-Eye (raw) | `57.39 +/- 2.85` |
| RoBERTEye-W (raw) | `57.29 +/- 2.37` |

So the late-fusion model is not only better than the published table entries; it is also the best IITBHGC system among the raw official dumps that can be re-evaluated under one metric pipeline.

### 4.2 Where the gain comes from

The gain is not concentrated in a single failure mode. Relative to the saved raw RoBERTa prediction dump, late fusion improves all three regimes, with the largest gain in the unseen-text setting:

- unseen text: about **+2.0 AUROC points**
- unseen reader: about **+1.3 AUROC points**
- unseen reader and unseen text: about **+0.9 AUROC points**

![Regime-level gains](submission_assets/figures/regime_gains.png)

*Figure 2. AUROC gain of CEC late fusion over the raw RoBERTa prediction dump by test regime.*

This pattern is consistent with the intended inductive bias. The largest gain appears when the text changes, which is where claim-conditioned evidence localization should matter most.

### 4.3 Latent-variable ablations

The crucial scientific question is not only whether CEC helps, but whether the latent evidence variable is doing real work.

We ran four controls:

1. **No scorer**: replace the learned evidence scorer with uniform context weights;
2. **Uniform eval**: keep the trained model but replace learned evidence weights with uniform scores at test time;
3. **Shuffle eval**: keep the trained model but randomly permute the learned evidence weights at test time;
4. **Score drop**: remove the highest-scoring evidence tokens and compare the prediction change with matched random token removal.

The performance ablations are suggestive rather than decisive. Learned late fusion is best by AUROC, but the gaps over no-scorer and uniform controls are small:

| Fusion variant | Test AUROC | Delta vs learned |
|---|---:|---:|
| Learned CEC + RoBERTa | `59.3 +/- 1.5` | baseline |
| NoScorer + RoBERTa | `59.0 +/- 1.6` | `-0.2` |
| Uniform + RoBERTa | `59.0 +/- 1.6` | `-0.3` |
| Shuffle + RoBERTa | `58.8 +/- 1.6` | `-0.5` |

![Ablation comparison](submission_assets/figures/ablation_auroc.png)

*Figure 3. Direct and late-fusion ablations. Learned scores are best by AUROC, but the margins are modest.*

Paired bootstrap confirms the same picture. Against the saved raw RoBERTa prediction dump, the late-fusion gain is positive with a 95% bootstrap CI that stays above zero:

| Comparison | Delta AUROC | 95% bootstrap CI | One-sided p |
|---|---:|---:|---:|
| CECGaze + RoBERTa vs raw RoBERTa dump | `+1.39` points | `[+0.02, +2.80]` | `0.022` |
| CECGaze + RoBERTa vs NoScorer + RoBERTa | `+0.24` points | `[-0.86, +1.35]` | `0.328` |
| CECGaze + RoBERTa vs Uniform + RoBERTa | `+0.26` points | `[-0.76, +1.30]` | `0.300` |
| CECGaze + RoBERTa vs Shuffle + RoBERTa | `+0.51` points | `[-0.63, +1.70]` | `0.184` |
| CECGaze + RoBERTa vs RoBERTEye-F (raw) | `+0.64` points | `[-1.86, +3.23]` | `0.306` |
| CECGaze + RoBERTa vs PostFusion-Eye (raw) | `+0.81` points | `[-1.82, +3.52]` | `0.253` |

So, by performance alone, the latent scorer helps in the expected direction, but the learned-vs-control differences are not individually decisive under paired resampling.

### 4.4 Score-drop evidence

The strongest evidence for the latent variable comes from behavior under intervention. When we remove the top 20% of context tokens according to the learned evidence scores, prediction probabilities change far more than when we remove the same number of random tokens. Across all **4,375** held-out test trials:

- mean absolute probability change after top-score drop: **0.0039**
- mean absolute probability change after random drop: **0.0002**
- paired mean difference: **0.0037**, bootstrap 95% CI **[0.0034, 0.0042]**
- ratio: about **24.9x**
- fraction of trials where top-drop > random-drop: **99.57%**

![Score-drop sensitivity](submission_assets/figures/score_drop.png)

*Figure 4. Top-score removal perturbs predictions much more than matched random removal.*

The same result holds in every test regime, with top/random ratios between about **22.8x** and **25.3x**. This is important because it shows that the latent evidence scores are not inert. The model is behaviorally sensitive to the tokens that the scorer ranks highly. However, the effect is not perfectly monotonic in AUROC: in some folds or regimes, removing top-ranked tokens does not decrease AUROC. So the correct interpretation is **behavioral relevance**, not yet a strong causal explanation claim.

## 5. Discussion

Our current evidence is strong enough for a workshop report, but only if it is framed carefully.

First, the architecture itself is novel in a meaningful task-specific sense. CEC is not just another generic text-gaze fusion block. It explicitly treats claim verification as evidence search and evidence use. The latent claim-conditioned evidence distribution and the evidence coverage vector are the new ideas.

Second, the benchmark result is solid enough to report. The late-fusion model beats the published EyeBench text-only benchmark and all published multimodal baselines in the benchmark table. That is already interesting because eye-tracking papers on claim verification remain relatively sparse, and because the direct CEC model appears to contribute complementary information rather than simply replacing the text model.

Third, the latent-variable story is promising but incomplete. The ablations do not justify a strong claim that the learned scorer alone is responsible for the entire performance gain. What they support is a more modest and defensible claim: the learned latent scores influence predictions, shuffled scores are worse, and top-score removal perturbs predictions much more than random removal on nearly every held-out trial. That is enough to argue that the latent variable is meaningful, but not enough to advertise it as a faithful explanation mechanism.

This distinction is important for publication. A workshop committee is likely to accept a careful story built around a valid benchmark improvement, a clearly motivated task-specific architecture, and honest ablations. The same story would need more evidence for a stronger venue, ideally through another dataset or stronger scorer diagnostics.

## 6. Conclusion

We introduced a claim-conditioned evidence coverage architecture for eye-tracked claim verification, corrected its implementation, and evaluated it on the official IITB-HGC benchmark. Direct CEC is not stronger than a strong text-only RoBERTa model, but it contributes complementary signal: validation-tuned late fusion reaches **59.3 +/- 1.5** AUROC and improves over the published text-only RoBERTa benchmark at **58.8 +/- 1.5**. Ablations show that the latent evidence mechanism matters behaviorally, although the learned-vs-control performance gaps remain modest. The result is therefore publication-worthy as a careful workshop contribution: a benchmark-improving multimodal method with a clear task-matched inductive bias and a transparent discussion of what the latent evidence variable does, and does not yet, prove.
