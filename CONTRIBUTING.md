# Contributing to Entroptics Positivity

## Build and test

```bash
pip install -r research/requirements.txt
python -m pytest research/code/tests -q
```

**Every gate carries a negative control that makes it able to fail.** A new gate arrives with one,
because in this rig a defect does not look like a crash — it looks like a result.

Every threshold is derived from the arithmetic rather than chosen. A constant that had to be
invented gets swept, and the spread reported.

## Contributing

Fork, branch from `main`, sign off every commit (`git commit -s`) to certify the
[DCO](https://developercertificate.org/), open a PR. Commit format: `fix:` · `feat(scope):` ·
`docs:` · `test:` · `chore:`.

By contributing you agree your contribution is Apache-2.0 (per section 5), including its
section 3 patent grant.

Licensed under Apache-2.0 — see [`LICENSE.md`](LICENSE.md) and [`NOTICE`](NOTICE).
