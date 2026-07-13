# Reproducibility checklist

This repository provides the CHARM model components, configuration loading, performer-disjoint
split generation, reference-density calibration, CSCS computation, and executable training and
evaluation entry points. It does not include third-party motion recordings or private run
artifacts.

For a fully reproducible result release, add the following files without changing their
contents:

1. `data/*_splits.json` generated from the released sequence identifiers.
2. One frozen YAML configuration per released run.
3. Model checkpoints or persistent archival links, if dataset licenses allow them.
4. Per-seed metric JSON files used to construct aggregate tables.
5. A machine-readable environment lock file produced on the training platform.
6. The authorized non-heritage initialization checkpoint or its full training procedure.

Numerical results should not be regenerated from illustrative or randomly constructed inputs.
Unit tests use small tensors only to verify software behavior; they are not experimental
observations.

Exact reproduction requires the frozen pose detector, dataset-specific joint map, pretraining
checkpoint, and other parameters from the original experiment environment. These details must
be recovered from that environment rather than guessed during release preparation.
