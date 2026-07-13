# Contributing

Please open an issue before making a substantial change. A contribution should include a
focused description, tests for changed behavior, and any necessary documentation updates.

Development checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python scripts/audit_release.py
```

Do not submit third-party recordings, derived pose archives whose redistribution is not
permitted, participant identifiers, credentials, trained weights without provenance, or
values reconstructed from illustrative data. Report security-sensitive problems privately
to the repository owner rather than in a public issue.
