# PARSE22 independent supervised protocol

## Scope and frozen split

This experiment family is **PARSE22 training → PARSE22 validation → PARSE22 testing**. It is independent of the historical **AirRC M0 → PARSE22 public100-case cross-dataset evaluation** retained in `reports/parse2022/`. Those cross-dataset scores are not P0 or a row in the new comparison.

The100 publicly labeled cases are split by published patient-case ID into64 training,16 validation and20 testing cases using NumPy PCG64 `default_rng(2026)`, a permutation of the sorted100 IDs, then sorted lists within each partition. The frozen list is [parse22_seed2026.json](../data/splits/parse22_seed2026.json); SHA-256 is `b1fc47e360539f00b8a12bb756f8caf4ec916c0658e0d5117fdd39544d1b3a36`. There is no cross-ID patient linkage in the public release; exact decoded CT duplicate checking supplements case IDs. No duplicate scans were found. No difficult cases were removed or resampled.

The pulmonary [VesMamba repository](https://github.com/Lzpbright/VesMamba), main commit `d491834149b0aa399deb4bf7ee9f5ca3a2d13475`, exposed a README without a concrete case split or implementation at lookup. The project therefore uses its own fixed split.64/16/20 describes the experiment on the public100 labeled cases, not the entire challenge's official partition. Paper numbers are literature references; they do not establish a strictly fair comparison. A direct implementation comparison would require running official code under this same frozen protocol.

All100 cases were previously evaluated with frozen AirRC M0. The new20-case test set is a holdout for this supervised experiment, **not a study-wide never-viewed blind test set**. Since freezing this split, test results are not used to alter data, algorithms, hyperparameters, thresholds or checkpoint selection.

## Input and physical space

Source CT/label pairs were checked for case pairing, decoded CT duplicates, shapes, spacing, affine/origin/direction, finite raw HU, binary integer labels, nonempty foreground and readable volumes. Native labels are **0 background /1 pulmonary artery**. The AirRC raw-label mapping is never applied. There is no complete vein or airway GT, so this family computes no vein/airway scores, MCS, A→V/V→A or merged A/V metrics.

Eleven cases triggered a conservative foreground low-HU heuristic: PA000056,128,133,148,150,233,237,296,307,309,310. Three native axial CT/GT overlays per case were reviewed. No gross global displacement was observed; peripheral/partial-volume annotation concerns remain possible. The original labels and every case were retained. This visual sample is not a claim of complete anatomical annotation verification.

A consistency check during engineering smoke found abnormal decoded HU in the **old server copy of PA000157 CT** while the checked local original remained within−1024…3071 HU. The new supervised source manifest uses a fresh original-file copy for this case. Training-only intensity statistics and all affected normalized caches were regenerated before formal training. No GT or split changed. The old cross-dataset inputs/results were retained; the currently damaged cached file alone cannot establish whether the historical inference input was already affected. Historical scores were not silently recalculated.

Only the64 training cases fit the fingerprint, foreground CT intensity statistics and automated planning. Validation16 is transformed with frozen training plans; test20 has no training cache or fit role. CT uses cubic continuous interpolation and labels use nearest-neighbor order0, including label spatial augmentation. Binary logits/probabilities use the existing inverse preprocessing/export into the **original GT physical grid**. Size, spacing, origin, direction and binary labels are checked after export. Array-size-only resizing is not used.

The training64 native XYZ spacing ranges from approximately(0.553,0.553,1.0) to(0.924,0.924,1.0)mm, with median(0.652,0.652,1.0)mm. Eight training cases spanning native voxel volumes were used in a predefined nearest-neighbor GT roundtrip retention check.1mm retained mean94.09% of native GT skeleton voxels (minimum90.79%);0.8mm retained97.09% (minimum95.12%). The predeclared rule selects the coarsest candidate among1.0/0.8/0.75mm with mean≥95% and minimum≥90%, so **0.8mm isotropic** is fixed for all three groups. This is an engineering representation check using training GT, not a model score or a test-driven choice. Detailed cases and values are saved in `training_resolution_study.json`.

## Experimental matrix and training

| Group | Backbone and loss | Added module |
|---|---|---|
| P0 | Binary U-Mamba bottleneck3D, existing foreground Dice+CE | None |
| P1 | P0 plus Skeleton Recall Loss, weight1 | None |
| P2 | Exactly the P1 loss | Supplied CT-guided residual refiner |

Every formal group starts from scratch; no AirRC, trained P0 or trained P1 weights initialize another group. Backbone weights are initialized before the new refiner, compared through a shared initial tensor digest, and data RNG is reset after initialization. Shared seed2026, augmentation, foreground sampling,128³ patch, batch2, SGD(lr0.01,momentum0.99,Nesterov,weight decay3e-5), polynomial schedule and gradient clipping12 are common. Resolved augmentation/optimizer/deep-supervision snapshots are saved. The deterministic single-process augmentation iterator uses the same initial sampling stream across groups. This does not claim all custom CUDA kernels are bitwise deterministic.

Each group receives **100000 successful optimizer updates**,400 epochs×250 updates. AMP overflow retries reuse the same batch and do not count as updates. All groups use the same policy. The budget is a project configuration, not an exact reproduction of VesMamba's paper schedule.50-update engineering smoke runs are isolated and discarded as formal initialization. All three interfaces are checked before P0 formal training; the formal order is P0, then P1, then P2.

Skeleton Recall Loss adapts only the mathematical loss and target transform from [MIC-DKFZ/Skeleton-Recall](https://github.com/MIC-DKFZ/Skeleton-Recall), commit `c7bac3c72ca7bb4f3d891235f051f0c4d07d7229`, with Apache2 attribution in `third_party/skeleton_recall/`. After spatial augmentation, Lee GT skeletons are dilated by2 voxels with Manhattan connectivity and intersected with GT. This training tube is distinct from the undilated evaluation skeleton. The loss is negative soft foreground recall, smoothing1e-5, with the baseline batch-Dice setting. Only the highest-resolution deep-supervision target has the extra tube channel; lower heads and the original deep-supervision weights are retained. The installed framework is not upgraded to a different augmentation stack.

## Supplied P2 module

[models/ct_local_refiner.py](../models/ct_local_refiner.py) implements the code supplied in [CT_Local_Refiner_Codex.md](CT_Local_Refiner_Codex.md), with framework `decoder`/`do_ds` forwarding. Inputs are normalized CT, detached full class probabilities and detached normalized entropy. Two3×3×3 Conv/GroupNorm/SiLU blocks use16 hidden channels, followed by a3×3×3 residual head. With entropy H, the gate is `0.25 +0.75H` and correction is `gate*tanh(raw_delta)` with `max_delta=1.0`. Head weights/bias start at zero; the initial refined logits equal backbone logits exactly.

The refiner runs in FP32 inside mixed-precision training. It changes only highest-resolution logits(index0); other deep-supervision outputs remain unchanged. There is no added residual regularizer. All refiner parameters enter the optimizer, and complete wrapper weights are saved and loaded. Training and inference both enable the module **on each normalized patch before Gaussian sliding-window fusion**. Feature gradients may initially be zero because the output head is zero; they must become nonzero after the head updates. Delta absolute mean/p50/p90/p99, parameter count, memory and timing are recorded. If needed, refiner-only activation checkpointing is the first memory remedy; common patch/batch settings are not silently reduced.

The convolutional path has a7×7×7 receptive field, but GroupNorm uses spatial statistics, so the whole module is **not a strictly finite-neighborhood operator**. The three-group comparison does not isolate CT-input contribution or standalone refinement efficacy and cannot by itself establish module complementarity.

## Validation, inference and saved checkpoints

Every2000 updates, all16 frozen validation patients receive complete volume inference. The only selection score is mean per-patient pulmonary-artery Dice; ties retain the earlier checkpoint. Auxiliary topology metrics never replace the selection criterion. Latest checkpoints are saved every250 successful updates after required validation finishes, with optimizer/scheduler/AMP/RNG/scope state. Resumption is restricted to the same group, split and formal run kind. Checkpointing before an incomplete full validation cannot silently skip that selection event.

Inference uses128³ windows, tile step0.5(50% overlap), Gaussian weighting, no TTA and no postprocessing. All three groups share these settings; the P2 refiner is the declared experiment variable. Test20 is evaluated only after all three frozen training trajectories finish. Each selected model also has validation16 evaluated separately. Original-grid predictions, inference timing/settings and checkpoint provenance are saved per case. Existing predictions can be reused for later metric calculation without another inference pass.

## Metrics and statistics

Base scores are Dice, IoU, clDice, HD95(mm), Precision and Recall, reusing the existing implementations and empty-mask conventions. **clDSC(%) = clDice×100**. Supplementary Skeleton Recall/Precision, BDR80, BDR90 and branch coverage reuse the unchanged prior branch implementation and cached native-grid GT branches. The recorded prior numeric-code digests must match.

Skeleton recall is GT skeleton inside prediction; skeleton precision is predicted skeleton inside GT. No dilation or physical tolerance is added. Branches are maximal GT skeleton paths between endpoint/junction nodes with26-neighborhood connectivity and contracted adjacent junction clusters. Branch lengths use physical affine distances, junction-internal edges are excluded, and the fixed3mm minimum filters short branches with explicit exclusion counts. Pure cycles and degenerate structures are recorded. Edge half-length weights define same-class prediction coverage along each GT branch. BDR80 is the fraction with coverage≥80%; BDR90 is sensitivity analysis only. Branches are equal-weight within each patient; patients are macro-averaged. Coverage-based detection does **not** guarantee branch continuity.

Primary results are20 test patients' individual metrics followed by mean, sample SD(ddof=1), median and n_valid. Validation statistics remain separate. NaNs are not replaced by zero; exceptions and case IDs are retained. P1−P0, P2−P0 and P2−P1 pair exact case IDs and use10000 patient-level paired percentile bootstrap replicates with seed2026 for95% CIs. Precision and Recall must be interpreted together. One training seed is the initial screen; patient bootstrap does not replace training-seed variability. The final P0/method comparison should subsequently add multiple training seeds.

Each group produces the requested eight primary files plus supporting protocol, history, initialization and exception records. Prediction/weight files are persisted privately, while public code/configuration/report copies exclude raw medical images, private connection details and machine-specific absolute paths. No score is populated until it was actually measured.
