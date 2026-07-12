# Security policy

## CI gates

GitHub Actions use immutable action revisions, least-privilege read-only
permissions, and full-history Gitleaks secret scanning. GitHub dependency
review is included as an opt-in pull-request gate: enable dependency graph and
the needed repository security setting, then set the `DEPENDENCY_REVIEW_ENABLED`
repository variable to `true`. These gates complement, but do not replace,
review of model and dataset artifacts before running research workloads.

## Responsible disclosure

Do not open a public issue for a potential credential leak or unsafe artifact
deserialization path. Contact the repository owner privately with the affected
commit, impact, and reproducible details. Do not attach real credentials,
proprietary game assets, or private datasets.
