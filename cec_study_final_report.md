# CEC EyeBench Final Report

Last updated: 2026-04-12

## Executive Summary

We revisited the EyeBench CEC implementation, fixed several issues in how claim/context inputs and latent evidence scoring were wired, and reran the study on the official IITBHGC 4-fold setting with a 10-epoch `RoBERTa-Large` CEC backbone.

The strongest result is:

> `CECGaze + RoBERTa` late fusion reaches `59.3 +/- 1.5` average test AUROC, which is above the published EyeBench `Text-Only Roberta` benchmark at `58.8 +/- 1.5`.

That is the clean main claim.

The more careful secondary claim is:

> The latent scorer appears behaviorally relevant, but the current evidence is stronger for "these scores affect predictions" than for "these scores cleanly identify monotonic supportive evidence."

## Experimental Setting

- Dataset: IITBHGC
- Evaluation protocol: official EyeBench 4-fold CV with the 3 standard test regimes
- Main CEC model: `CECGaze`
- CEC text backbone: `RoBERTa-Large`
- Training budget: `10` epochs
- Main comparison target: published EyeBench `Text-Only Roberta`
- Fusion method: fold-local late fusion between official text-only RoBERTa predictions and CEC predictions
- Fusion weight selection: validation AUROC only
- Balanced-accuracy thresholding for supplementary reporting: validation-tuned threshold per fold and regime

## What We Corrected Before Running The Study

We did not treat the original CEC implementation as automatically trustworthy. Before the final runs, we verified and corrected the following:

1. Claim/context tokenization and masking:
   - fixed paired-input construction
   - fixed claim/context mask handling
   - fixed downstream inversion logic

2. Model configuration:
   - switched the CEC backbone to `RoBERTa-Large`
   - ran the study at `10` epochs
   - enabled the intended unfrozen setting for the improved model

3. Ablation support:
   - added `CECGazeNoScorer`
   - added score evaluation controls for learned, uniform, and shuffled latent scores

4. Evaluation reliability:
   - added cache isolation support
   - added MPS-aware evaluation fallback
   - added score-drop evaluation controls and batch-size overrides

## Final Results

### Primary benchmark comparison

The official benchmark comparison should stay AUROC-first.

| Model | Test AUROC | Test BA | Notes |
|---|---:|---:|---|
| Text-Only Roberta (published EyeBench benchmark) | `58.8 +/- 1.5` | `52.5 +/- 1.4` | official reference table |
| CECGaze direct | `56.1 +/- 0.3` | `54.6 +/- 0.5` | BA uses validation-tuned threshold |
| CECGaze + RoBERTa late fusion | `59.3 +/- 1.5` | `56.8 +/- 0.8` | best AUROC |

Important note:

- The `56.8 +/- 0.8` balanced-accuracy number for late fusion uses validation-tuned thresholds.
- The published EyeBench benchmark table does not report our threshold-tuning protocol, so AUROC remains the cleanest official apples-to-apples metric.
- We should therefore present the AUROC win as the main benchmark result, and the threshold-tuned balanced accuracy as a secondary descriptive metric.

### Late-fusion ablation table

| Fusion variant | Test AUROC | Test BA (val-tuned) | Read |
|---|---:|---:|---|
| CECGaze + RoBERTa | `59.3 +/- 1.5` | `56.8 +/- 0.8` | best AUROC |
| NoScorer + RoBERTa | `59.0 +/- 1.6` | `57.5 +/- 1.3` | slightly lower AUROC, higher BA |
| Uniform-score + RoBERTa | `59.0 +/- 1.6` | `57.1 +/- 1.3` | below learned AUROC |
| Shuffled-score + RoBERTa | `58.8 +/- 1.6` | `56.9 +/- 1.3` | worst AUROC, roughly baseline-level |

Interpretation:

- Learned late fusion is the best model by AUROC.
- The gains over no-scorer and uniform are real but modest.
- Shuffling the latent scores hurts most clearly.
- The latent scorer currently helps the ranking story more cleanly than the thresholded classification story.

### Direct CEC ablation table

| Direct variant | Test AUROC | Test BA (val-tuned) |
|---|---:|---:|
| CECGaze | `56.1 +/- 0.3` | `54.6 +/- 0.5` |
| CECGazeNoScorer | `56.1 +/- 0.4` | `52.9 +/- 0.8` |
| CECGaze with uniform scores | `55.9 +/- 0.3` | `53.5 +/- 0.7` |
| CECGaze with shuffled scores | `55.6 +/- 0.2` | `54.1 +/- 0.8` |

Interpretation:

- Direct CEC does not beat the strong text-only benchmark.
- The learned scorer is not dramatically better than the no-scorer control in direct AUROC.
- The direct model story alone is therefore not strong enough to carry the paper.

## How To Present The Balanced-Accuracy Story

Do not use the fixed-`0.5` balanced accuracy from the raw late-fusion summary as the headline. The fused probabilities are not calibrated for a universal `0.5` operating point, which is why the naive summary collapses to `50.0 +/- 0.0`.

The cleaner wording is:

> We treat AUROC as the primary benchmark metric. For balanced accuracy, we additionally report a validation-tuned decision threshold per fold and regime, because the fused scores are not calibrated to a fixed `0.5` threshold.

That gives us a sane secondary metric without pretending it is the core benchmark target.

## Latent Scorer Evidence

### What the ablations support

There is some evidence that the latent scorer matters:

- learned fusion is best by AUROC
- shuffle hurts more than the other controls
- score-drop interventions affect predictions much more than matched random dropping

### Score-drop result

Setup:

- remove the top `20%` of context tokens under the learned evidence weights
- compare against matched random token removal
- use `10` random repeats

Across folds:

| Score-drop summary | Value |
|---|---:|
| Mean abs probability change after top-drop | `0.00380 +/- 0.00242` |
| Mean abs probability change after random-drop | `0.000159 +/- 0.000086` |
| Top-drop / random-drop change ratio | `18.1x +/- 5.0x` |
| Mean removed evidence mass | `0.337 +/- 0.072` |
| Mean dropped tokens | `24.21 +/- 0.07` |

What this supports:

- the latent scorer is behaviorally active
- the tokens it ranks highly are not interchangeable with random tokens

What this does **not** support cleanly:

- a simple claim that removing top-ranked evidence always degrades AUROC

In some folds or regimes, AUROC after top-drop did not decrease monotonically. So the careful interpretation is:

> The scorer identifies behaviorally influential regions, but it is not yet proven to be a clean monotonic selector of supportive evidence for the correct class.

That is still publishable, but it needs to be framed honestly.

## Recommended Claim Wording

If we want the paper pitch to stay strong without overselling, I would phrase it like this:

> After correcting the original CEC implementation and rerunning the IITBHGC benchmark under the official 4-fold setting, we find that CEC-derived gaze predictions improve over the published text-only RoBERTa baseline when combined through validation-tuned late fusion, reaching `59.3 +/- 1.5` average AUROC versus the benchmark `58.8 +/- 1.5`. Ablations show that shuffled latent scores are consistently weaker, and score-drop interventions change predictions far more than matched random token removal, suggesting that the latent scorer is behaviorally meaningful even though its causal interpretation remains imperfect.

## What We Should Claim, And What We Should Not

### Safe claims

- We corrected meaningful implementation issues in the original CEC path.
- The corrected CEC + text late-fusion model beats the published text-only RoBERTa benchmark on average AUROC.
- The latent scorer is not inert; it measurably affects predictions.
- Shuffled latent scores weaken results.

### Claims to avoid

- "Direct CEC beats text-only RoBERTa"
- "The latent scorer is definitively causal"
- "Eye tracking alone solves the task better than text"
- "The method is clearly superior across every metric and ablation"

## Publication Read

Most realistic current framing:

- good workshop paper
- good short paper
- good benchmark / replication / implementation-correction note

Less secure, unless we extend the study:

- full paper at a stronger venue based only on the current gap

The current story becomes much stronger if we add at least one of:

- another dataset
- a stronger calibration analysis
- more robust scorer diagnostics
- a stronger end-to-end multimodal model that still preserves the late-fusion result

## Key Artifact Paths

- Working log:
  - `cec_study_progress.md`
- Final fusion summary:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeRobertaValBlendFine/summary/summary_metrics.csv`
- Final fusion threshold summary:
  - `source/eyebench/outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeRobertaValBlendFine/summary/threshold_summary_metrics.csv`
- Direct CEC summary:
  - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/summary/local_benchmark_summary_metrics.csv`
- Direct CEC threshold summary:
  - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/summary/local_benchmark_threshold_summary_metrics.csv`
- Score-drop summaries:
  - `source/eyebench/outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGaze/fold_index=*/score_drop_summary.csv`

## Bottom Line

The result worth showing off is real:

> corrected CEC + late fusion beats the published text-only EyeBench RoBERTa benchmark on AUROC.

The result worth presenting carefully is also real:

> the latent scorer matters, but the current ablations support a behavioral interpretation more strongly than a clean causal-evidence interpretation.
