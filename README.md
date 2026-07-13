# CHARM reference implementation

This repository contains a clean-room reference implementation of **Canonical
Heritage-Aware Representation and Motion modeling (CHARM)** and the **Canonical Style
Centrality Score (CSCS)** described in:

> You Fu. *Canonical Style beyond Realism: Disentangled, Low-Resource Modeling of
> Intangible Heritage Performance.* Information Sciences submission.

CSCS is a corpus-relative, sample-level density rank. It does not measure cultural validity
or generative distribution coverage.

## What is included

- performer-disjoint train/calibration/test split generation;
- skeleton-graph/temporal-transformer style--execution encoder and mirrored decoder;
- gradient-reversed category/performer adversaries;
- cross-performer supervised contrastive and swap-cycle losses;
- category-specific eight-layer RealNVP reference priors;
- held-out empirical-CDF calibration and CSCS evaluation;
- category-restricted top-k retrieval memory;
- retrieval-conditioned motion diffusion with classifier-free guidance;
- YAML configurations matching the principal hyperparameters reported in the paper;
- tests and GitHub Actions configuration.

The third-party datasets, unpublished run logs, and trained checkpoints are not included.
No experimental table is reconstructed from invented or illustrative observations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Python 3.10--3.12 and PyTorch 2.2 or newer are supported.

## Data preparation

Download the datasets from their original providers:

- [Chang-E Dunhuang dance corpus](https://cislab.hkust-gz.edu.cn/projects/chang-e/)
- [Balinese basic-movement video corpus](https://doi.org/10.17632/s2gv9d6gpb.2)

The repository does not bypass their access conditions or redistribute recordings. For the
video corpus, calibrated 2D detections can first be triangulated with
`scripts/triangulate_multiview.py`. Convert licensed local 3D arrays into the format in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md):

```bash
python scripts/prepare_archive.py \
  --motions local/motions.npy \
  --categories local/categories.npy \
  --performers local/performers.npy \
  --sequence-ids local/sequence_ids.npy \
  --output data/processed/dunhuang_motions.npz \
  --manifest data/processed/dunhuang_manifest.json \
  --source "Chang-E, locally downloaded version" \
  --source-url "https://cislab.hkust-gz.edu.cn/projects/chang-e/" \
  --download-date "YYYY-MM-DD" \
  --license-note "Record the applicable provider access terms here" \
  --target-frames 196

charm-make-splits \
  --archive data/processed/dunhuang_motions.npz \
  --output data/dunhuang_splits.json \
  --seed 7
```

Whole performers are assigned to one partition, so the 60/15/25 sequence ratios are
targets rather than exact guarantees. The preparation command root-centers at joint 0 by
default and records every requested transform in a JSON manifest. Use `--joint-map` and
`--scale-bone` only when they match the actual corpus processing. The joint map and skeleton
edges must use the same target ordering.

## Training and evaluation

```bash
charm-train-representation \
  --config configs/dunhuang.yaml \
  --output checkpoints/dunhuang_representation.pt

charm-fit-reference \
  --config configs/dunhuang.yaml \
  --representation checkpoints/dunhuang_representation.pt \
  --output checkpoints/dunhuang_reference.pt

charm-train-generator \
  --config configs/dunhuang.yaml \
  --representation checkpoints/dunhuang_representation.pt \
  --output checkpoints/dunhuang_generator.pt

charm-evaluate \
  --config configs/dunhuang.yaml \
  --representation checkpoints/dunhuang_representation.pt \
  --reference checkpoints/dunhuang_reference.pt \
  --output artifacts/dunhuang_test_cscs.json
```

If an authorized non-heritage pretraining checkpoint is available, pass it through
`--initialize-from`. This repository does not include or manufacture the manuscript's
pretraining checkpoint.

Use `--epochs` on the training commands for a short integration check. The paper
configurations retain the reported 1200 representation epochs, 800 generator epochs, 1000
diffusion steps, latent dimensions of 128, retrieval `k=8`, and guidance weight `w=3`.

## Reproducibility boundary

The public code supplies executable method components and deterministic split logic. Exact
paper-table reproduction additionally requires the authors' frozen split files, checkpoints,
per-seed outputs, and dataset-specific pose preprocessing details. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before creating a release.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source repository is
[vectorlab5/CHARM](https://github.com/vectorlab5/CHARM); a final article DOI can be added after
publication.

Before publishing, follow [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) and run:

```bash
pytest
python scripts/audit_release.py
```

## License

A software license has intentionally not been selected yet. The repository owner should add
the chosen license before making the GitHub repository public.
