# Attribution and rights

## Original competition research

The original report, **2024 APMCM Problem A**, was written collaboratively by the competition team. The team has agreed to publication of the report in this repository. The report is a historical research artifact and is **not licensed under MIT** by the software license in the root `LICENSE` file. Do not infer permission to relicense, modify, or commercially redistribute the report from the software license.

The repository maintainer may add an author list or an approved citation once all contributors agree on the public wording. Until then, credit the **original APMCM competition team** rather than attributing the team report to a single individual.

## Reconstructed software

The newly authored software in this repository is licensed under the root MIT `LICENSE`, subject to any separately identified third-party notices. This reconstruction uses YOLOv5s; the original report discusses YOLOv5x. Their training results must not be conflated.

## Third-party material

- YOLOv5 and ZoeDepth are external upstream projects. They are not vendored here and retain their own licenses and attribution requirements.
- Pretrained weights, original competition photographs and training datasets are not included in the tracked repository.
- The report contains citations and figures sourced or adapted from third parties. Team consent alone does not settle rights in third-party material. Check figure-level permissions and competition terms before switching the repository to public.
- Generated evaluation charts and metrics are reported as comparisons against historical **unreviewed pseudo-labels**, not independent ground truth.

## Research limitations

Maturity cluster labels are a user-defined interpretation of unsupervised HSV groups. ZoeDepth results can be inaccurate on orchard scenes. The mass output is an uncalibrated proxy in arbitrary units, **not grams**.
