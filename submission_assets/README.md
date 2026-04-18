# Submission Asset Summary

## Key Numbers

- Raw text-only RoBERTa AUROC: 57.9 +/- 2.1
- Best direct CEC variant (CECGazeNoCoverage) AUROC: 60.8 +/- 1.8
- Best late-fusion CEC variant (NoCoverage+RoBERTa) AUROC: 61.5 +/- 1.3

## Re-aggregated Raw Official Baselines

| model                   |   auroc_mean |   auroc_sem |   balanced_accuracy_val_tuned_mean |   balanced_accuracy_val_tuned_sem |
|:------------------------|-------------:|------------:|-----------------------------------:|----------------------------------:|
| Text-Only Roberta (raw) |       0.5789 |      0.0215 |                             0.5500 |                            0.0175 |

## Bootstrap Comparisons

| model_a         | model_b                 |   delta_mean |   delta_sem_boot |   delta_ci_low |   delta_ci_high |   p_one_sided |   p_two_sided |
|:----------------|:------------------------|-------------:|-----------------:|---------------:|----------------:|--------------:|--------------:|
| CECGaze+RoBERTa | Text-Only Roberta (raw) |       0.0227 |           0.0002 |         0.0016 |          0.0451 |        0.0150 |        0.0300 |
| CECGaze+RoBERTa | NoScorer+RoBERTa        |      -0.0029 |           0.0002 |        -0.0184 |          0.0132 |        0.3475 |        0.6950 |
| CECGaze+RoBERTa | Uniform+RoBERTa         |       0.0000 |           0.0000 |        -0.0002 |          0.0003 |        0.5135 |        1.0000 |
| CECGaze+RoBERTa | Shuffle+RoBERTa         |      -0.0000 |           0.0000 |        -0.0003 |          0.0002 |        0.3440 |        0.6880 |
| CECGaze         | CECGazeNoScorer         |      -0.0021 |           0.0002 |        -0.0194 |          0.0159 |        0.3895 |        0.7790 |
| CECGaze         | CECGazeUniformEval      |      -0.0001 |           0.0000 |        -0.0004 |          0.0001 |        0.2225 |        0.4450 |
| CECGaze         | CECGazeShuffleEval      |      -0.0001 |           0.0000 |        -0.0005 |          0.0002 |        0.2680 |        0.5360 |
| CECGaze+RoBERTa | NoCoverage+RoBERTa      |      -0.0129 |           0.0002 |        -0.0272 |          0.0013 |        0.0400 |        0.0800 |
| CECGaze+RoBERTa | TextOnly+RoBERTa        |       0.0169 |           0.0003 |        -0.0061 |          0.0396 |        0.0735 |        0.1470 |
| CECGaze         | CECGazeNoCoverage       |      -0.0092 |           0.0002 |        -0.0244 |          0.0071 |        0.1290 |        0.2580 |
| CECGaze         | CECGazeTextOnly         |       0.0205 |           0.0003 |        -0.0082 |          0.0489 |        0.0770 |        0.1540 |

## Score-Drop Trial Statistics

| eval_regime                |   n_trials |   mean_abs_delta_top_drop |   mean_abs_delta_random_drop |   mean_abs_delta_difference |   difference_ci_low |   difference_ci_high |   p_one_sided |   p_two_sided |   top_over_random_ratio |   fraction_top_gt_random |   mean_removed_top_evidence_mass |   mean_n_dropped_tokens |
|:---------------------------|-----------:|--------------------------:|-----------------------------:|----------------------------:|--------------------:|---------------------:|--------------:|--------------:|------------------------:|-------------------------:|---------------------------------:|------------------------:|
| average                    |       7500 |                    0.0371 |                       0.0357 |                      0.0014 |              0.0009 |               0.0019 |        0.0000 |        0.0000 |                  1.0392 |                   0.5415 |                           0.2181 |                 24.2220 |
| seen_subject_unseen_item   |       3126 |                    0.0332 |                       0.0321 |                      0.0011 |              0.0004 |               0.0018 |        0.0004 |        0.0008 |                  1.0355 |                   0.5317 |                           0.2183 |                 24.2415 |
| unseen_subject_seen_item   |       3124 |                    0.0411 |                       0.0396 |                      0.0015 |              0.0008 |               0.0022 |        0.0000 |        0.0000 |                  1.0374 |                   0.5435 |                           0.2180 |                 24.2266 |
| unseen_subject_unseen_item |       1250 |                    0.0370 |                       0.0352 |                      0.0019 |              0.0007 |               0.0030 |        0.0002 |        0.0004 |                  1.0526 |                   0.5608 |                           0.2178 |                 24.1616 |

## Figures

- `figures/benchmark_auroc.png`
- `figures/ablation_auroc.png`
- `figures/regime_gains.png`
- `figures/score_drop.png`

