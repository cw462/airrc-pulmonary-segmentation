# AirRC M0 / P1 / P2



M0 (P0_M0) is the completed four-label U-Mamba baseline. P1 adds Skeleton Recall Loss for all three foreground classes. P2 adds the user supplied CT guided residual refiner to P1.

P1 and P2 start independently from scratch on separate GPUs. PARSE22 supervised training is paused by user instruction; its saved checkpoint is retained.



The original fixed178/38/38 split,1mm preprocessing,128³ patch,batch2,4 training/2 validation augmentation workers and100000 stock AMP training iterations are reused. Successful optimizer updates and AMP skips are recorded separately; the historical M0 skip count is unavailable.

Full validation38 every2000 iterations selects the mean patient Dice across PA/PV/airway lumen. Test38 is excluded from selection. M0 scores are imported unchanged from reports/airrc_m0.

Basic, skeleton/branch and AV metrics retain the frozen project protocols. clDSC(%) equals clDice×100. BDR80 measures GT branch length coverage, not continuous connection.



Each summary uses patient mean, sample_std(ddof=1), median and n_valid, separately for validation and test. Ratios are stored0–1; HD95 is mm. Paired bootstrap uses seed2026 and10000 resamples.

One training seed does not establish robustness to training randomness. These three groups do not establish module complementarity or refinement alone.



Current pipeline state: **training**.

User requested pause: **P2**. GPU1 is reserved for the user’s other experiments. P1 continues; P2 will resume only after explicit user instruction.
