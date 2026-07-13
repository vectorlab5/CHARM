# GitHub release checklist

Complete this checklist before making the repository public.

- [ ] Confirm the project name, version, and anonymous attribution in `pyproject.toml`.
- [ ] Select and add a `LICENSE`; do not assume that dataset licenses cover this software.
- [ ] Confirm each dataset URL, version, download date, and license/access note.
- [ ] Verify the joint map, skeleton edges, root joint, scale convention, and sequence length
      against the preprocessing used for the intended workflow.
- [ ] Add frozen performer-disjoint split JSON files if their identifiers may be redistributed.
- [ ] Add per-seed configurations, environment details, and checksum records for any released
      checkpoints or result files.
- [ ] Run `pytest` and `python scripts/audit_release.py` from a clean checkout.
- [ ] Check that no source recordings, credentials, local absolute paths, editor metadata, or
      private notes are committed.
- [ ] Create a versioned release and, when useful, archive it with a persistent identifier.

Until the frozen splits, preprocessing choices, checkpoints, and per-seed outputs are added,
describe this repository as an executable implementation rather than a complete artifact
reproducing every archived result.
