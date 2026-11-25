# Canonical benchmark driver

This fork's `bench/` directory holds campaign/job configurations. The generic
measurement driver is **not** here — it lives once in `luni-base`:

- Driver: `/srv/luni-repos/luni-base/scripts/bench/bench.sh`
- Method, test matrix, upstream comparability, c1..cN and prefill data:
  `luni-base/docs/benchmarks.md`
- Provenance rules: `luni-base/standards/startup-provenance.md`

```bash
cd /srv/luni-repos/luni-base
./scripts/bench/bench.sh --all --label <new>          # after every restart
./scripts/bench/bench.sh baseline --repeat 3          # upstream method, medians
./scripts/bench/bench.sh --prefill-repeat 12 cold-prefill
./scripts/bench/bench.sh --ramp-levels 1,2,3,4 ramp   # or 4,5,6 as overload probe
```

`--all` = doctor → probe → acc → bb → cold-prefill → prefill → agent → ramp →
telemetry. `bb --repeat N` emits `BB-AGG` min/med/max for every c-level and
prefill depth; every run writes `benchmark-identity.json` (commit SHAs,
effective flags, recipe hash, qualification, artefact hashes).

Results: `/srv/llm-cache/qwen38-flashnext/ab-night/<label>/`. Default target is
the unmodified donor on :8001; use `--url` / `--model` for a Speedy build.
Only same-method numbers are comparable; warm-prefix prefill is not a prefill
compute measurement. `hyperburst-rtxpro6000` is a **reference** repo only — do
not add benchmark code there.
