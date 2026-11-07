# AirRC M0 weights

The selected checkpoint is from epoch 400, iteration 100,000. It predicts background, pulmonary artery, pulmonary vein and airway lumen.

Download checkpoint_best.pth from the v0.1.0 release. The file is the full training checkpoint with model, optimizer and resume state. [Release](https://github.com/cw462/airrc-pulmonary-segmentation/releases/tag/v0.1.0)

Place the file in fold_0/checkpoint_best.pth under a model directory containing configs/nnUNetPlans.json as plans.json and configs/dataset.json as dataset.json. The evaluator accepts that model directory through --model.
