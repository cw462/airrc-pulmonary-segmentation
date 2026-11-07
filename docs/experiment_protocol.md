# Experiment protocol

## AirRC M0

The experiment trains a three-dimensional U-Mamba bottleneck model from scratch using AirRC. Patient splitting uses seed 2026 with 178 training, 38 validation and 38 held-out test patients. The released split file lists the public LIDC-IDRI identifiers.

The original CT is linearly resampled into the official 1 mm AirRC annotation grid using its physical geometry. Resampled HU values are rounded to int16. The label conversion follows the table below.

| Original AirRC label | Structure | Model label |
|---|---|---|
| 0 | Background | 0 |
| 1 | Airway lumen | 3 |
| 2 | Airway wall | 0 |
| 3 | Pulmonary artery | 1 |
| 4 | Pulmonary vein | 2 |

The frozen nnU-Net plan uses CT normalization fitted from the 178 training patients. The actual clipping percentiles, mean and standard deviation are recorded in configs/nnUNetPlans.json. Preprocessed images are cached as float16 and converted to float32 for model input.

The patch size is 128 × 128 × 128 with physical batch size 2. Training uses 400 epochs with 250 update iterations per epoch, giving 100,000 iterations. The optimizer is SGD with initial learning rate 0.01, momentum 0.99, Nesterov momentum and weight decay 0.00003. The framework's polynomial learning-rate schedule, foreground Dice plus cross-entropy loss, deep supervision and augmentation pipeline are retained.

Full-volume validation runs every 2,000 iterations on the same 38 patients. The checkpoint with the highest three-class mean validation Dice is selected. Patch-based pseudo Dice and its moving average are diagnostic training curves. This run uses one random seed.

## Inference

Inference uses the frozen training normalization, 1 mm processing spacing, 128-cube windows, 50% overlap, Gaussian blending and a single prediction pass. Predictions are restored to the reference image grid through the saved image geometry.

AirRC evaluation uses the official 1 mm annotation grid. PARSE2022 evaluation uses the original CT and label grid. PARSE2022 pulmonary artery predictions are obtained by taking label 1 from the original three-class argmax output.

## PARSE2022 external evaluation

The external cohort consists of all 100 publicly released, labeled PARSE2022 training cases. Their role in this project is external evaluation of the AirRC-trained M0. The model, normalization and inference parameters are fixed using the AirRC experiment. The 100 cases are assigned across separate GPU workers for inference.

The target is pulmonary artery. The official label convention is background 0 and pulmonary artery 1. The case list is recorded in data/splits/parse2022_public_labeled_cases.txt. The local manifest builder creates paths for an independently obtained copy of the data.

## Metrics

Metrics are calculated per patient and summarized with the arithmetic mean and sample standard deviation.

- Dice, IoU, precision and recall use the binary target for each evaluated structure.
- clDice uses 3D Lee skeletonization and the harmonic mean of centerline precision and sensitivity.
- HD95 is the maximum of the two directed 95th-percentile surface distances. Surface points use six-connected boundary voxel centers, and distances use the original physical spacing in millimeters.
- Both-empty masks are excluded from metric means and counted separately. A single empty mask receives zero overlap and clDice scores, with the physical image diagonal used as the HD95 penalty.

## Method and data sources

- AirRC dataset and annotation protocol, Scientific Data, 2025. https://doi.org/10.1038/s41597-025-06074-6
- AirRC author repository. https://github.com/b-niu/AirRC
- U-Mamba backbone and legacy nnU-Net implementation. https://github.com/bowang-lab/U-Mamba
- PARSE2022 dataset documentation. https://parse2022.grand-challenge.org/Dataset/
- clDice, CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html

