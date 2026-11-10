# AirRC M0/P1/P2 experiment protocol

The user moved the primary experiment back to AirRC. Existing M0 is reused as
P0_M0. PARSE22 supervised P0 was saved at1750 successful optimizer updates and
paused; its data, split, configuration and checkpoint are retained separately.

|Group|Model/loss|Initialization|
|---|---|---|
|P0_M0|Completed U-Mamba baseline, original segmentation loss|Existing completed experiment|
|P1|Same baseline plus Skeleton Recall Loss on PA/PV/airway lumen|Independent from scratch|
|P2|P1 plus user supplied CT guided local residual refiner|Independent from scratch|

The fixed AirRC split is178 training,38 validation and38 testing patients in
`data/splits/airrc_seed2026.json`. Labels are0 background,1 pulmonary artery,
2 pulmonary vein,3 airway lumen. The prepared GT already uses this encoding;
the original raw mapping[0,3,0,1,2] must not be applied a second time.

P1/P2 reuse the exact frozen M0 plan and normalization fitted on training178:
1mm isotropic,128³ patch,batch2, four-label output. No refitting or replanning.
They retain M0's original augmentation,4 training/2 validation background
workers, foreground oversampling0.33, SGD(lr0.01,momentum0.99,Nesterov,
weight_decay3e-5), polynomial400-epoch schedule, gradient clipping12 and
original Dice+CE deep supervision weights. Seed2026 and identical initial
backbone tensors are verified between P1/P2. Historical M0 initial tensors
were not saved; their exact identity cannot be retrospectively verified.
M0's original nondeterministic background batch arrival is retained; a worker
restart cannot reproduce the interrupted sequence bit for bit.

The common budget is400×250=100000 training iterations with the original M0
AMP behavior. New runs separately record actual successful optimizer updates
and AMP skips. M0 did not record this distinction; no claim is made that it
completed100000 *successful* updates. P1/P2 each use one3090, batch2, noDDP.
Parallel dispatch changes scheduling only. Other jobs are allowed to finish.

On2026-11-08 the user reserved GPU1 for other experiments. P2 was paused at
the8500-iteration checkpoint(8497 successful updates plus3AMP skips), while
the existing P1 process continued on GPU0. The dispatcher honors a persistent
user pause flag and cannot automatically start P2 even if GPU1 becomes idle.
Restoring P2 requires explicit user instruction. This scheduling change does
not alter the split, scientific configuration, saved state or training budget.

Full validation uses all38 patients every2000 iterations. The selection score
matches M0: patient mean Dice for each of the three classes, then mean across
classes. The highest score wins, with earlier checkpoint retained on ties.
Test38 never participates in selection. M0's completed test has already been
examined in this study; it is not a new study-wide blind test.

Skeleton Recall Loss uses Lee skeletons, a2-voxel Manhattan tube intersected
with the same-class GT after spatial augmentation, highest-resolution target
only, unit loss weight and equal foreground-class averaging. It is added
inside the unchanged highest-resolution deep-supervision weight. Its signed
negative-recall form and smoothing1e-5 follow the documented implementation;
it is not a four-class background recall objective.

P2 uses `models/ct_local_refiner.py` supplied by the user, now configured for
four output classes: hidden16,max_delta1,gate_floor0.25,FP32 under AMP,
zero-initialized residual head, detached probability/entropy conditions and
highest-resolution logits only. Training and inference both use the complete
wrapper; refinement occurs before Gaussian sliding-window fusion. There is
no additional residual regularization. GroupNorm aggregates spatial
statistics, so the convolutional7³ receptive field does not imply strictly
local statistics. P1/P2 never load trained M0 or smoke checkpoints as their
formal initial weights.

Inference retains128³ windows,50% overlap, Gaussian weighting, noTTA and no
postprocessing. Outputs are restored through the existing physical-space
export to the fixed M0 evaluation GT grid. Geometry and label checks precede
voxel comparisons. Basic metrics reuse `evaluate_airrc.py`; skeleton/branch
metrics reuse the frozen GT graphs and protocol:26 adjacency, junction
clusters, minimum branch length3mm,0mm tolerance, physical-length coverage,
BDR80 primary andBDR90 sensitivity. Coverage does not prove continuity.
clDSC(%) means clDice×100. AV MCS is
`(A→V_voxels + V→A_voxels)/(GT_A_or_V_voxels + Pred_A_or_V_voxels)`,
with no factor2. No identity with unverified HiPaS author code is claimed.

Metrics are computed per patient, then summarized separately for validation
and test by class(mean, sample_std withddof1, median,n_valid). NaN is never
zero-filled. AV metrics use one row per patient, not three repeated class
rows. Comparisons pair by the fixed case ID, use seed2026 and10000 patient
bootstrap resamples, and report mean differences and percentile95% CIs.
Patient bootstrap does not replace multiple training seeds. P1/P2/M0 alone
do not establish module complementarity or refinement alone.

Engineering checks include three-class targets/loss gradients, four-class
zero-initialized refinement,250 smoke iterations including200 timed fresh
batches, wrapper checkpoint reload, and whole-volume physical-grid export.
Every training snapshot records its own family/group/split/run kind.
Resuming another group or initializing formal training from smoke is refused.

The server's shared PyTorch2.1.0 CUDA library was found inconsistent with the
original package record during launch. A task-specific overlay restores the
same original2.1.0/CUDA12.1 package library without modifying the shared
environment or changing versions. The time of historical corruption is
unknown. A CUDA/Mamba witness and full smoke runs are required before use.
Public reports contain code/configuration/metrics only; images, labels,
predictions, weights and private connection details remain outside Git.

On 2026-11-10, after P1 completed its fixed 100000-iteration budget, the user
requested P2 and all subsequent training/evaluation to use the same GPU0.
P2 resumes its own saved 8500-iteration checkpoint, including optimizer,
scheduler, AMP scaler and RNG state, with the same configuration and output
directory. Remaining jobs share GPU0 serially; GPU1 remains reserved for
the user's other experiments. This changes device placement only, with no
new M0/P1 training and no changes to the frozen scientific recipe. The
original multiworker batch-arrival semantics remain nondeterministic.
