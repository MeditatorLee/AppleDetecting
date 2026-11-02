# Data and evaluation provenance

- Attachment 1: 200 source photographs; historical CSV contains 2,570 predicted boxes.
- Dataset split: 140 train / 29 validation / 31 test images, 1,748 / 449 / 373 pseudo-labeled boxes.
- These are historical **model-generated predictions**, not independently reviewed ground-truth annotations.
- The dataset split tool grouped exact duplicate image hashes, but near-duplicate/related image leakage was not independently ruled out.
- Test run reported precision 0.872, recall 0.784, mAP@50 0.859, mAP@50–95 0.543. Results are agreement with pseudo-labels.
- The original competition report, training results CSV and images are preserved for provenance. Current checkpoint is stored locally and not included.
- Use `scripts/check_environment.py` to record runtime versions and Git revisions before future experiments.
- For physical mass accuracy, collect camera calibration/reference scale and measured fruit masses, then evaluate error on an independent set.

## Publication checklist

Before uploading: confirm dataset image rights, review any original report team attribution, remove personal filesystem paths from configs and screenshots, verify no `.pt` files or raw archives are staged, and publish a separate model card if distributing weights.
