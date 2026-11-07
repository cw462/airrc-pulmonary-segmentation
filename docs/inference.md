# 对新 CT 使用已有 AirRC M0

`inference.predict_frozen` 显式读取已有 checkpoint，对指定 CT 推理并导出原始 CT 网格的标签。训练归一化、模型参数、128-cube 窗口、50% overlap、Gaussian blending 和无 TTA 设置沿用原模型。内部使用既有 `evaluate_airrc.predict_case` 及 nnU-Net 空间还原，原 CT 文件不被修改。

manifest 是一份独立 JSON，例如：

~~~json
{
  "cases": [
    {"case_id": "example1", "image_path": "/data/example1/ct.nii.gz"}
  ]
}
~~~

在已配置 U-Mamba 的 Linux 环境运行：

~~~bash
export PYTHONPATH="$PWD:$UMAMBA_ROOT"
export nnUNet_raw=/path/to/run/full/raw
export nnUNet_preprocessed=/path/to/run/full/preprocessed
export nnUNet_results=/path/to/run/full/results

python -m inference.predict_frozen \
  --manifest /path/to/ct_manifest.json \
  --model /path/to/run/full/results/Dataset926_AirRC/AirRCTrainer__nnUNetPlans__3d_fullres \
  --checkpoint-epoch 400 --output /path/to/new_predictions --gpus 0 1
~~~

每例产生 `predictions/CASE_ID_model_label.nii.gz` 和 `results/CASE_ID.json`。标签为背景 0、肺动脉 1、肺静脉 2、气道管腔 3。记录包含原始 CT 与预测的 shape、spacing、origin、direction、affine、类别体素数和推理时间，导出时检查物理网格一致。

这个命令没有 GT 输入，不计算准确率。TotalSegmentator 等其他模型的输出可以作为自动分割对照保存，不能代替完整人工动静脉 GT 计算 AV 分离准确率。AV 补充评价仍使用独立的 [固定 GT 评分脚本](av_metrics.md)。
