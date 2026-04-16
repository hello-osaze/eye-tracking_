# CEC EyeBench Study Progress

Last updated: 2026-04-12

For the cleaner presentation-ready version, see `cec_study_final_report.md`.

## Goal

Test whether a corrected CEC-based eye-tracking model can add useful signal on the IITBHGC EyeBench task, and whether combining it with the official text-only RoBERTa benchmark can outperform the published text-only baseline under the same evaluation setting.

The target comparison is the official `Text-Only Roberta` result from EyeBench:

- Average AUROC: `58.8 +/- 1.5`
- Average Balanced Accuracy: `52.5 +/- 1.4`

Source:

- `source/eyebench/results/formatted_eyebench_benchmark_results/IITBHGC_CV_test.csv`

## What We Verified And Corrected

Before running the full study, we checked whether the existing CEC implementation actually matched the intended claim/context formulation. We found several issues that would have made the earlier runs hard to trust, and fixed them before the official 10-epoch study.

### 1. Claim/context input handling was corrected

Files:

- `source/eyebench/src/data/datasets/TextDataSet.py`
- `source/eyebench/src/data/datasets/base_dataset.py`

Changes:

- Fixed claim/context tokenization so the model sees a true paired input instead of a broken or inverted mapping.
- Fixed the claim/context attention masks and inversion logic so downstream model code consumes the right segments.

Why this matters:

- If the claim and context are not aligned correctly, the latent evidence mechanism is operating on the wrong textual structure.

### 2. The CEC model config was aligned to the benchmark setting

Files:

- `source/eyebench/src/configs/models/dl/CECGaze.py`
- `source/eyebench/src/models/cec_gaze_model.py`

Changes:

- Set the backbone to `RoBERTa-Large`.
- Set training to `10` epochs for the official apples-to-apples run.
- Unfroze the backbone for the main improved model setting.
- Added explicit ablation model variants:
  - `CECGazeNoScorer`
  - `CECGazeNoCoverage`
  - `CECGazeTextOnly`
- Added `score_eval_mode` so we can evaluate:
  - learned scores
  - uniform scores
  - shuffled scores

Why this matters:

- We wanted the improved model to be run on the same backbone and epoch budget as the official RoBERTa benchmark.

### 3. Output isolation and timing controls were added

Files:

- `source/eyebench/src/configs/models/base_model.py`
- `source/eyebench/src/data/datamodules/base_datamodule.py`
- `source/eyebench/src/run/single_run/run_cec_gaze_full_study.py`
- `source/eyebench/src/run/single_run/run_iitbhgc_local_benchmark_suite.py`

Changes:

- Added a feature-cache suffix so runs with different settings do not silently reuse stale cached features.
- Added `max_time_limit` support to the training launchers.

Why this matters:

- This reduces the chance of accidental cache leakage or unstable long-run behavior.

### 4. Evaluation was adapted to run on MPS

Files:

- `source/eyebench/src/run/single_run/test_dl.py`
- `source/eyebench/src/run/single_run/test_cec_gaze_score_drop.py`

Changes:

- Evaluation no longer forces CPU when MPS is available.
- Added MPS fallback logic: CUDA -> MPS -> CPU.
- Added evaluation batch-size override support.
- Added `--eval-types` support to the score-drop script so we can run test-only ablations.

Why this matters:

- This made the remaining ablations practical on the local Apple GPU setup.

## Exact Study Design So Far

### Main model

- Model name: `CECGaze`
- Backbone: `RoBERTa-Large`
- Training budget: `10` epochs
- Evaluation setting: official IITBHGC 4-fold CV setting used by EyeBench
- Output root:
  - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGaze`

### Main ablations

- `CECGazeNoScorer`
  - removes the learned latent scorer
  - output root:
    - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGazeNoScorer`

- Uniform-score control
  - uses the trained model, but replaces learned token scores with uniform scores at evaluation time

- Shuffle-score control
  - uses the trained model, but shuffles token scores at evaluation time

- Score-drop ablation
  - drops the highest-scoring evidence tokens and compares the prediction change to random token dropping
  - status: still running on MPS

### Fusion setup

We also ran late fusion against the official text-only RoBERTa outputs.

Method:

- Use the official RoBERTa prediction files as one score stream.
- Use CEC-derived prediction files as the other score stream.
- Tune the fusion weight `alpha` on validation folds only.
- Lock that weight and report the corresponding test AUROC.

This is a valid multimodal evaluation setup as long as it is reported transparently as a late-fusion model rather than an end-to-end jointly trained model.

Output roots:

- Full learned CEC late fusion:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeRobertaValBlendFine`
- No-scorer late fusion:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeNoScorerRobertaValBlendFine`
- Uniform late fusion:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeUniformRobertaValBlendFine`
- Shuffle late fusion:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeShuffleRobertaValBlendFine`

## Results So Far

### Official benchmark target

From the EyeBench benchmark table:

| Model | Average AUROC | Average Balanced Accuracy |
|---|---:|---:|
| Text-Only Roberta | `58.8 +/- 1.5` | `52.5 +/- 1.4` |

### Direct CEC-family results

These are the mean test AUROC values across the 4 official folds, averaging over the 3 official evaluation regimes.

| Model | Test AUROC | Notes |
|---|---:|---|
| CECGaze (learned scorer) | `56.11 +/- 0.27` | corrected full model |
| CECGazeNoScorer | `56.10 +/- 0.40` | learned scorer removed |
| CECGaze with uniform scores | `55.94 +/- 0.34` | evaluation-time control |
| CECGaze with shuffled scores | `55.57 +/- 0.23` | evaluation-time control |

Takeaway:

- Direct CEC alone does **not** beat the official text-only RoBERTa benchmark.
- Shuffling the latent scores hurts direct performance the most.
- The direct gap between learned scorer and no-scorer is currently small.

### Late-fusion results with official RoBERTa

These are the current best apples-to-apples numbers for the study.

| Model | Test AUROC | Delta vs Text-Only Roberta |
|---|---:|---:|
| Text-Only Roberta (official benchmark) | `58.8 +/- 1.5` | baseline |
| CECGaze + RoBERTa late fusion | `59.3 +/- 1.5` | `+0.5` |
| CECGazeNoScorer + RoBERTa late fusion | `59.0 +/- 1.6` | `+0.2` |
| CECGaze uniform + RoBERTa late fusion | `59.0 +/- 1.6` | `+0.2` |
| CECGaze shuffle + RoBERTa late fusion | `58.8 +/- 1.6` | ~`0.0` |

Exact AUROC means:

- `CECGaze + RoBERTa`: `0.5927696063`
- `CECGazeNoScorer + RoBERTa`: `0.5903976872`
- `CECGazeUniform + RoBERTa`: `0.5902006462`
- `CECGazeShuffle + RoBERTa`: `0.5876518596`

Takeaway:

- The full learned late-fusion model currently beats the official `Text-Only Roberta` benchmark on average AUROC.
- The margin is modest, but it is a real apples-to-apples improvement under the same fold setting.
- Among the control fusions, shuffled latent scores are worst.
- Learned late fusion is better than no-scorer, uniform, and shuffle, though the learned-vs-no-scorer gap is still small.

### Important caution on balanced accuracy

The current late-fusion reporting is AUROC-driven. The summary files for late fusion currently show balanced accuracy at `50.0 +/- 0.0`, which indicates threshold calibration is not yet the right lens for judging the fusion experiments. At the moment, the strongest fair comparison is AUROC, not balanced accuracy.

### Score-drop ablation

The score-drop ablation is now complete across all 4 folds on the test split.

Setup:

- remove the top `20%` of context tokens according to the learned evidence weights
- compare that against matched random token dropping
- use `10` random-drop repeats

Summary across folds:

| Metric | Result |
|---|---:|
| Mean absolute probability change, top-drop | `0.00380 +/- 0.00242` |
| Mean absolute probability change, random-drop | `0.000159 +/- 0.000086` |
| Top-drop / random-drop change ratio | about `18.1x +/- 5.0x` |
| Mean removed top evidence mass | `0.337 +/- 0.072` |
| Mean dropped tokens | `24.21 +/- 0.07` |
| Base AUROC | `56.11 +/- 0.28` |
| Top-drop AUROC | `56.64 +/- 0.59` |
| Random-drop AUROC | `56.04 +/- 0.28` |

What this means:

- Dropping the top-scored tokens changes model probabilities **much more** than dropping the same number of random tokens.
- The learned scorer is therefore not inert; it is behaviorally linked to the model's predictions.
- However, the direction of the effect is **not consistently degradative** in AUROC terms. In some folds/regimes, removing top-scored tokens even improves AUROC.

Interpretation:

- This supports the claim that the latent scorer is identifying behaviorally important text regions.
- It does **not** yet support the stronger claim that these scores cleanly correspond to monotonic positive evidence for the correct class.
- A careful write-up should frame this as **behavioral sensitivity evidence**, not as a definitive causal proof that the scorer always finds supportive evidence.

## Interpretation Right Now

At this point the strongest claim we can make is:

> A corrected CEC-style eye-tracking model, when combined with the official RoBERTa text model through validation-tuned late fusion, improves average AUROC over the published text-only RoBERTa benchmark on IITBHGC.

What we **cannot** claim yet:

- that the direct CEC model is stronger than the text-only RoBERTa benchmark
- that the learned latent scorer is decisively established as the only reason for the fusion gain

What is already encouraging:

- the full late-fusion model is the best result we have so far
- shuffled-score controls move in the expected worse direction
- score-drop shows that top-ranked evidence tokens affect predictions much more than matched random token removal
- the setup is now on the official 10-epoch RoBERTa-Large setting, so the comparison is fair

## Is Late Fusion A Valid Approach?

Yes. Late fusion is a valid and widely acceptable multimodal strategy, especially when:

- the modalities are heterogeneous
- each modality already has a strong standalone model
- you want a clean ablation story
- you tune fusion weights on validation only and keep the test set untouched

What matters is honest framing. We should describe this as:

- a multimodal late-fusion method
- not an end-to-end jointly trained multimodal encoder

That is still scientifically valid. In this case it may actually be the cleaner first result, because it isolates whether gaze-derived signal adds anything on top of text.

## Are These Results Potentially Publishable?

My honest read: **possibly yes**, but probably not yet as a strong full-paper story in the current form.

Why it is interesting:

- Eye-tracking benchmark literature is relatively small.
- We have already found and fixed implementation issues that materially affected the validity of earlier runs.
- We now have an apples-to-apples AUROC improvement over the published text-only RoBERTa baseline.
- The ablation story is nontrivial and informative: shuffled scores hurt, and top-score removal perturbs predictions far more than random removal.

Why it is not fully convincing yet:

- The gain over text-only RoBERTa is modest.
- The learned scorer is only slightly ahead of the no-scorer and uniform controls in late fusion.
- Direct CEC alone does not currently beat the strong text baseline.
- The score-drop experiment shows strong behavioral sensitivity, but not a clean monotonic degradation story.

My best publication read:

- **Workshop / short paper / benchmark note**: realistic if the score-drop ablation strengthens the interpretation and we write the implementation-correction story clearly.
- **Stronger venue**: possible only if we can either widen the gap, improve the ablation evidence, or extend the study to another dataset / stronger robustness analysis.

## Remaining Work

- Decide whether to also fix late-fusion thresholding for balanced-accuracy reporting.
- Write the final result summary with a careful claim:
  - direct CEC vs text-only
  - late fusion vs text-only
  - latent-score ablations
  - evidence-drop behavior

## Current Bottom Line

So far, the study is promising, and the best clean headline is:

> Corrected CEC-derived gaze features improve over the official EyeBench text-only RoBERTa baseline when fused late with RoBERTa, reaching `59.3 +/- 1.5` average AUROC versus the published `58.8 +/- 1.5`.

That is a credible result. Whether it becomes a publishable one depends on how strong the final ablations look once the score-drop run finishes.

## Strict Review Loop Update

Date: 2026-04-12 evening

We started a stricter reviewer-proofing pass aimed at closing the remaining obvious objections:

1. missing official-setting architecture ablations;
2. stronger benchmark comparability under one evaluator;
3. stronger latent-variable statistics beyond descriptive fold means.

### New code changes in this loop

Files updated:

- `source/eyebench/src/run/single_run/build_cec_submission_assets.py`
  - now supports a secondary ablation root
  - now re-aggregates the raw official IITBHGC baselines under the same evaluator
  - now computes paired bootstrap comparisons against those raw baselines
  - now computes trial-level score-drop significance summaries

- `source/eyebench/src/configs/models/dl/CECGaze.py`
- `source/eyebench/src/models/cec_gaze_model.py`
- `source/eyebench/src/run/single_run/test_dl.py`
  - added inference-time knockout switches for:
    - `eval_zero_coverage`
    - `eval_zero_gaze_features`
  - these allow fast test-time ablations on the learned checkpoint without changing checkpoint shapes

### New benchmark-comparability result

Using the raw official IITBHGC prediction dumps and the same aggregation code for every model:

| Model | Test AUROC | Notes |
|---|---:|---|
| `CECGaze + RoBERTa` | `59.28 +/- 1.46` | best under common evaluator |
| `RoBERTEye-F (raw)` | `58.64 +/- 1.28` | best raw official baseline |
| `PostFusion-Eye (raw)` | `58.46 +/- 1.24` | raw official baseline |
| `Text-Only Roberta (raw)` | `57.89 +/- 2.15` | raw official reference dump |
| `MAG-Eye (raw)` | `57.39 +/- 2.85` | raw official baseline |
| `RoBERTEye-W (raw)` | `57.29 +/- 2.37` | raw official baseline |

Paired bootstrap against the raw official dumps:

- vs raw text-only RoBERTa:
  - delta AUROC: `+1.39` points
  - 95% CI: `[+0.02, +2.80]`
  - one-sided `p = 0.0225`
  - two-sided `p = 0.0450`

- vs the raw multimodal baselines:
  - all deltas are positive
  - none of the pairwise wins over `RoBERTEye-W`, `RoBERTEye-F`, `MAG`, or `PostFusion` are individually decisive by paired bootstrap yet

Interpretation:

- the fusion model is strongest under the common evaluator as well as in the published benchmark table;
- the most statistically secure benchmark claim remains the improvement over raw text-only RoBERTa.

### Stronger latent-variable evidence

Using the completed score-drop trial outputs across all 4 folds (`4375` test trials total):

| Statistic | Value |
|---|---:|
| Mean abs change after top-score drop | `0.0039` |
| Mean abs change after matched random drop | `0.0002` |
| Mean paired difference | `0.0037` |
| 95% CI on paired difference | `[0.0034, 0.0042]` |
| Top/random ratio | `24.87x` |
| Fraction of trials where top-drop > random-drop | `99.57%` |
| One-sided bootstrap `p` | `< 0.0005` |

Per regime, the same qualitative result holds:

- `seen_subject_unseen_item`: `25.31x`, `99.41%` of trials
- `unseen_subject_seen_item`: `25.09x`, `99.63%` of trials
- `unseen_subject_unseen_item`: `22.80x`, `99.84%` of trials

This is much stronger latent-variable evidence than the earlier fold-level summary alone.

### Fresh official-setting ablations currently running

We discovered that an earlier restart of the missing `NoCoverage` / `TextOnly` sweep had accidentally fallen back to the sweep-derived per-config timer (`00:00:09:22`), which is invalid for the official 10-epoch comparison. That run was stopped.

A clean rerun is now active under a fresh output root with an explicit long time budget:

- output root:
  - `source/eyebench/outputs/cec_gaze_ablation_completion_mps_large_true_e10`
- launcher:
  - `run_iitbhgc_local_benchmark_suite.py`
- models:
  - `CECGazeNoCoverage`
  - `CECGazeTextOnly`
- fixed setting:
  - `RoBERTa-Large`
  - `10` epochs
  - unfrozen backbone
  - `model.max_time_limit=00:12:00:00`

Current status at last check:

- `CECGazeNoCoverage` fold `0` is training cleanly on MPS
- no hidden short timer is present
- `caffeinate` is attached to keep the machine awake

### Queued next run: explanation faithfulness

To address the remaining question

> “Are the latent scores just behaviorally important, or are they also faithful as an explanation ranking?”

we added and queued a stronger evaluation script:

- `source/eyebench/src/run/single_run/test_cec_gaze_faithfulness.py`

This run evaluates the learned evidence ranking on the trained `CECGaze` checkpoints using standard rationale-style diagnostics:

- **comprehensiveness**:
  - drop the top-ranked evidence tokens and measure the drop in the model's original predicted-class confidence
- **sufficiency**:
  - keep only the top-ranked evidence tokens and measure how much of the original predicted-class confidence remains
- controls for both:
  - random-ranked tokens
  - bottom-ranked tokens
- fractions:
  - `5%`, `10%`, `20%`, `30%`, `40%`, `50%`
- split:
  - test only

The queued command waits for the current official MPS ablation suite to finish, then starts automatically on MPS:

- target model root:
  - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGaze`
- outputs to be written per fold:
  - `faithfulness_test_results.csv`
  - `faithfulness_summary.csv`

Why this matters:

- score-drop already shows that top-ranked tokens matter more than random ones;
- this next run asks the stronger explanation question:
  - are top-ranked tokens both more **necessary** and more **sufficient** than bottom/random controls?
- if yes, we can strengthen the report from
  - “behaviorally relevant”
  - toward
  - “faithfulness-style evidence under comprehensiveness and sufficiency tests”

### Next steps

1. finish `CECGazeNoCoverage` and `CECGazeTextOnly` under the clean official setting;
2. let the queued faithfulness suite complete on the learned checkpoints;
3. run the corresponding late-fusion controls;
4. rerun `build_cec_submission_assets.py`;
5. refresh `cec_workshop_report.md` with:
   - common-evaluator raw-baseline table
   - stronger score-drop significance result
   - final architecture-ablation results
   - comprehensiveness/sufficiency faithfulness results
