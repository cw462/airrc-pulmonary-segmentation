# PARSE22 independent supervised experiments

This family is separate from AirRC M0 → PARSE22 cross-dataset evaluation in `reports/parse2022/`.

Frozen patient-case split:64 train /16 validation /20 test; seed2026. Binary output:0 background,1 pulmonary artery. All formal groups start from scratch and use100000 successful optimizer updates.

Schedule: finish P0 on GPU0, then run P1 on GPU0 and P2 on GPU1 concurrently. Each model uses one RTX3090, batch2 and the same frozen training/validation conditions; this is independent-run parallelism, not DDP.

| Group | Method |
|---|---|
| P0 | U-Mamba Dice + CE baseline |
| P1 | P0 + Skeleton Recall Loss |
| P2 | P1 + supplied CT-guided residual refiner |

State: **training**. Pending summaries contain no measured scores.

| Group | Saved optimizer updates |
|---|---:|
| P0 | 1000 |
| P1 | 0 |
| P2 | 0 |

Engineering smoke runs are isolated interface checks, not formal model results or formal initialization.

Best checkpoint selection uses complete16-patient validation mean Dice every2000 updates. Test20 evaluation is deferred until all three frozen formal training trajectories finish.

Metrics use native GT physical grids and patient macro mean/sample SD(ddof=1)/median/n_valid. clDSC(%) is the same clDice value multiplied by100. BDR80 is the primary coverage-based branch detection metric; it does not prove continuous connection.

See [the protocol](../../docs/parse22_supervised_protocol.md) for geometry, train-only0.8mm planning, P2 architecture, prior100-case exposure, data-source correction, and interpretation limits.

Only code, configurations and reports are public. Source CTs, label volumes, prediction volumes, weights and private connection configuration are not copied by this report exporter.
