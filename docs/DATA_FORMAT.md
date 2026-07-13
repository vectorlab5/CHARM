# Data format

The repository does not redistribute the third-party Chang-E or Balinese recordings. Download
them from their original providers and follow their licenses.

Preprocessed motions are stored as a NumPy `.npz` archive with four arrays:

| Key | Shape | Type | Meaning |
|---|---:|---|---|
| `motions` | `(N, T, J, 3)` | `float32` | Root-normalized joint coordinates |
| `categories` | `(N,)` | `int64` | Contiguous movement-category identifiers |
| `performers` | `(N,)` | `int64` | Contiguous performer or take identifiers |
| `sequence_ids` | `(N,)` | string | Stable source-derived sequence identifiers |

Coordinates must use a common joint order and metric scale. Each sequence must contain only
one category and one performer identifier. Padding or resampling to a common length should be
recorded in a preprocessing manifest stored next to the archive.

## Preprocessing utilities

`scripts/prepare_archive.py` can reorder joints, root-center each frame, normalize scale by a
specified bone, and linearly resample time. It writes these choices into the provenance
manifest. Defaults are deliberately minimal: root-centering at joint 0 is enabled, whereas
joint remapping and scale normalization require explicit arguments.

For calibrated multi-view input, `scripts/triangulate_multiview.py` implements weighted DLT
triangulation from 2D detections and 3x4 camera projection matrices. Pose detection itself is
not prescribed by this repository. Record the detector name, version, weights, confidence
threshold, camera calibration, failed frames, and retargeting map alongside the generated
archive.

The 22-joint edge list in `configs/base.yaml` must be verified against the local joint map.
It is part of the model input definition, not a universally valid mapping for every provider.

## Split file

`charm-make-splits` writes JSON with `train`, `calibration`, and `test` sequence-ID lists.
Complete performer groups are allocated to one partition. Consequently, requested percentage
targets are approximate at sequence level.

## Required provenance

Before publishing an experimental release, retain the following outside Git when licensing
prevents redistribution:

- original dataset version and download date;
- joint mapping and coordinate normalization;
- excluded or failed sequences with reasons;
- generated split JSON;
- exact configuration file and random seed;
- checkpoint hashes and per-seed metric files.
