# GitHub release checklist

Complete this checklist before making the repository public.

- [ ] Confirm the author name and email in `CITATION.cff` and `pyproject.toml`.
- [ ] Select and add a `LICENSE`; do not assume that dataset licenses cover this software.
- [ ] Add the final GitHub URL to `CITATION.cff` and the manuscript Data Availability section.
- [ ] Confirm each dataset URL, version, download date, and license/access note.
- [ ] Verify the joint map, skeleton edges, root joint, scale convention, and sequence length
      against the actual preprocessing used for the reported experiments.
- [ ] Add frozen performer-disjoint split JSON files if their identifiers may be redistributed.
- [ ] Add per-seed configurations, environment details, and checksum records for any released
      checkpoints or result files.
- [ ] Run `pytest` and `python scripts/audit_release.py` from a clean checkout.
- [ ] Check that no source recordings, credentials, local absolute paths, editor metadata,
      review text, or manuscript comments are committed.
- [ ] Create a versioned release and archive it with a DOI-bearing service if required by the
      journal.

Until the frozen splits, preprocessing choices, checkpoints, and per-seed outputs are added,
describe this repository as an executable reference implementation rather than a complete
artifact reproducing every numerical table in the manuscript.
