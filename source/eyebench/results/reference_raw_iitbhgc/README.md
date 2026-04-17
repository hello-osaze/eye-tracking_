## IITBHGC Reference Raw Predictions

This folder contains the fold-level `trial_level_test_results.csv` files used as
the bundled Text-Only Roberta reference for the standalone CEC study.

Why this exists:

- the published EyeBench benchmark table gives us the official formatted summary
  rows, but late fusion and paired bootstrap comparisons need fold-level raw
  prediction files;
- a fresh single-config local rerun is not benchmark-equivalent to the original
  EyeBench sweep workflow;
- bundling these compact raw prediction files keeps the cloud pipeline aligned
  with the study reference instead of silently substituting a weaker fallback.

These files are intentionally limited to the IITBHGC Text-Only Roberta
reference used by this study.
