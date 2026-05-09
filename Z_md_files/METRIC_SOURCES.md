# Metric Sources

This repository's metric implementations are aligned with these public references:

- PRDC (Precision/Recall/Density/Coverage):
  - https://github.com/clovaai/generative-evaluation-prdc
- FID/IS/KID protocol references:
  - https://github.com/toshas/torch-fidelity
  - https://github.com/GaParmar/clean-fid
  - https://github.com/mseitzer/pytorch-fid

Notes:
- The local implementation extracts Inception-v3 features/probabilities directly and computes metrics in that feature space.
- Bootstrap confidence intervals are computed locally via percentile bootstrap in `Metrics/statistics.py`.
