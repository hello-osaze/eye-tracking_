# Submission Asset Summary

## Key Numbers

- Raw text-only RoBERTa AUROC: 57.9 +/- 2.1
- Direct CECGaze AUROC: 56.1 +/- 0.3
- Late fusion AUROC: 59.3 +/- 1.5

## Re-aggregated Raw Official Baselines

| model                   |   auroc_mean |   auroc_sem |   balanced_accuracy_val_tuned_mean |   balanced_accuracy_val_tuned_sem |
|:------------------------|-------------:|------------:|-----------------------------------:|----------------------------------:|
| Text-Only Roberta (raw) |       0.5789 |      0.0215 |                             0.5500 |                            0.0175 |
| RoBERTEye-W (raw)       |       0.5729 |      0.0237 |                             0.5452 |                            0.0207 |
| RoBERTEye-F (raw)       |       0.5864 |      0.0128 |                             0.5551 |                            0.0145 |
| MAG-Eye (raw)           |       0.5739 |      0.0285 |                             0.5595 |                            0.0245 |
| PostFusion-Eye (raw)    |       0.5846 |      0.0124 |                             0.5617 |                            0.0106 |

## Bootstrap Comparisons

| model_a         | model_b                 |   delta_mean |   delta_sem_boot |   delta_ci_low |   delta_ci_high |   p_one_sided |   p_two_sided |
|:----------------|:------------------------|-------------:|-----------------:|---------------:|----------------:|--------------:|--------------:|
| CECGaze+RoBERTa | Text-Only Roberta (raw) |       0.0139 |           0.0002 |         0.0002 |          0.0280 |        0.0225 |        0.0450 |
| CECGaze+RoBERTa | NoScorer+RoBERTa        |       0.0024 |           0.0001 |        -0.0086 |          0.0135 |        0.3275 |        0.6550 |
| CECGaze+RoBERTa | Uniform+RoBERTa         |       0.0026 |           0.0001 |        -0.0076 |          0.0130 |        0.3000 |        0.6000 |
| CECGaze+RoBERTa | Shuffle+RoBERTa         |       0.0051 |           0.0001 |        -0.0063 |          0.0170 |        0.1835 |        0.3670 |
| CECGaze         | CECGazeNoScorer         |       0.0001 |           0.0001 |        -0.0109 |          0.0117 |        0.4790 |        0.9580 |
| CECGaze         | CECGazeUniformEval      |       0.0017 |           0.0001 |        -0.0073 |          0.0112 |        0.3520 |        0.7040 |
| CECGaze         | CECGazeShuffleEval      |       0.0055 |           0.0001 |        -0.0043 |          0.0170 |        0.1470 |        0.2940 |
| CECGaze+RoBERTa | NoCoverage+RoBERTa      |       0.0028 |           0.0001 |        -0.0071 |          0.0123 |        0.2905 |        0.5810 |
| CECGaze+RoBERTa | TextOnly+RoBERTa        |       0.0139 |           0.0002 |         0.0002 |          0.0279 |        0.0225 |        0.0450 |
| CECGaze         | CECGazeNoCoverage       |       0.0024 |           0.0002 |        -0.0127 |          0.0164 |        0.3810 |        0.7620 |
| CECGaze         | CECGazeTextOnly         |       0.0772 |           0.0004 |         0.0453 |          0.1102 |        0.0000 |        0.0000 |
| CECGaze+RoBERTa | RoBERTEye-W (raw)       |       0.0199 |           0.0003 |        -0.0044 |          0.0443 |        0.0550 |        0.1100 |
| CECGaze+RoBERTa | RoBERTEye-F (raw)       |       0.0064 |           0.0003 |        -0.0186 |          0.0323 |        0.3055 |        0.6110 |
| CECGaze+RoBERTa | MAG-Eye (raw)           |       0.0189 |           0.0003 |        -0.0075 |          0.0464 |        0.0735 |        0.1470 |
| CECGaze+RoBERTa | PostFusion-Eye (raw)    |       0.0081 |           0.0003 |        -0.0182 |          0.0352 |        0.2525 |        0.5050 |

## Score-Drop Trial Statistics

| eval_regime                |   n_trials |   mean_abs_delta_top_drop |   mean_abs_delta_random_drop |   mean_abs_delta_difference |   difference_ci_low |   difference_ci_high |   p_one_sided |   p_two_sided |   top_over_random_ratio |   fraction_top_gt_random |   mean_removed_top_evidence_mass |   mean_n_dropped_tokens |
|:---------------------------|-----------:|--------------------------:|-----------------------------:|----------------------------:|--------------------:|---------------------:|--------------:|--------------:|------------------------:|-------------------------:|---------------------------------:|------------------------:|
| average                    |       4375 |                    0.0039 |                       0.0002 |                      0.0037 |              0.0034 |               0.0042 |        0.0000 |        0.0000 |                 24.8704 |                   0.9957 |                           0.3460 |                 24.2215 |
| seen_subject_unseen_item   |       1875 |                    0.0041 |                       0.0002 |                      0.0040 |              0.0032 |               0.0049 |        0.0000 |        0.0000 |                 25.3105 |                   0.9941 |                           0.3262 |                 24.2315 |
| unseen_subject_seen_item   |       1875 |                    0.0038 |                       0.0002 |                      0.0037 |              0.0033 |               0.0041 |        0.0000 |        0.0000 |                 25.0871 |                   0.9963 |                           0.3605 |                 24.2315 |
| unseen_subject_unseen_item |        625 |                    0.0035 |                       0.0002 |                      0.0033 |              0.0029 |               0.0037 |        0.0000 |        0.0000 |                 22.8036 |                   0.9984 |                           0.3617 |                 24.1616 |

## Figures

- `figures/benchmark_auroc.png`
- `figures/ablation_auroc.png`
- `figures/regime_gains.png`
- `figures/score_drop.png`
- `figures/cec_pipeline.png`
