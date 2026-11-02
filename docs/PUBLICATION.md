# Public release checklist

This checklist is for a **public GitHub release**, not a claim that all copyright or scientific questions are settled.

## Rights and provenance

- [x] The original team has agreed to publication of its report (reported by the repository maintainer).
- [ ] Confirm the competition's publication terms.
- [ ] Check each externally sourced or adapted figure in `docs/original_report.pdf` and replace, remove or obtain permission for material not cleared for public redistribution.
- [ ] Confirm the team's preferred public attribution and whether individual names should be listed.
- [x] Distinguish MIT-licensed reconstructed source from the original report and upstream model code.

## Repository hygiene

- [x] Ignore local datasets, checkpoints, caches, downloaded weights and external repositories.
- [ ] Run `git status` and `git ls-files` locally to ensure no unlicensed data or sensitive files are staged.
- [ ] Confirm that `docs/original_report.pdf` is tracked if the team wants it published. Undo any accidental local `git rm` before committing.
- [ ] Check for secrets, personal paths and metadata in committed files and historical Git commits before changing visibility.
- [ ] Verify `python -m pytest -q` and an end-to-end demo on the release commit.

## Versioning and publishing

- [x] `v1.0.1` annotated Git tag exists.
- [x] `main` points to the v1.0.1 release commit as checked on 2026-11-03.
- [ ] Merge the documentation PR after review; this will create a **post-v1.0.1** commit on `main`, not modify the immutable tag.
- [ ] Create a formal GitHub Release from the existing `v1.0.1` tag; a Git tag alone is not a GitHub Release entry.
- [ ] Review repository description, topics and public visibility only after rights checks.

## Reproducibility

The historical test metrics (precision 0.872, recall 0.784, mAP@50 0.859, mAP@50–95 0.543) measure agreement with pseudo-labels on 31 held-out images. They are **not independently validated detection accuracy**. The original competition's YOLOv5x and the reconstructed YOLOv5s are separate experiments.

Verified local compatibility during development: Python 3.11.18, PyTorch 2.7.1+cu128, `timm==0.6.13`, ZoeDepth commit `d87f17b2f5fdcb174cf4fb115491f4a6c60de152`. External dependencies and checkpoints must be obtained separately.
