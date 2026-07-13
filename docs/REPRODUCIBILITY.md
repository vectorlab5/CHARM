# Reproducibility checklist

This repository provides the CHARM model components, configuration loading, performer-disjoint
split generation, reference-density calibration, CSCS computation, and executable training and
evaluation entry points. It does not include third-party motion recordings or unpublished run
artifacts.

For a paper-result release, add the following files without changing their contents:

1. `data/*_splits.json` generated from the released sequence identifiers.
2. One frozen YAML configuration per reported run.
3. Model checkpoints or persistent archival links, if dataset licenses allow them.
4. Per-seed metric JSON files used to construct aggregate tables.
5. A machine-readable environment lock file produced on the training platform.
6. The authorized non-heritage initialization checkpoint or its full training procedure.

The numerical values in the article should not be regenerated from illustrative or randomly
constructed inputs. Unit tests use small tensors only to verify software behavior; they are not
experimental observations.

The executable implementation follows the disclosed graph-temporal architecture and reported
hyperparameters. Details absent from the manuscript—especially the frozen pose detector,
dataset-specific joint map, and pretraining checkpoint—must be recovered from the actual
experiment environment rather than guessed during release preparation.
